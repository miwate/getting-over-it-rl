using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using BepInEx.Logging;
using UnityEngine;

// Reads Unity's native Box2D world directly: what Physics2D's API can't return (the convex pieces Box2D simulates,
// contact manifolds with feature ids and warm-start impulses, sleep timers, joint impulses). Exact parity needs them.
// Nothing here is guessed blind: each b2Body is found by its pose (the bit pattern of Rigidbody2D.position followed by
// a unit rotation) in the pointer graph below the native Rigidbody2D, and every native read is first checked with
// VirtualQuery, so a wrong offset gives a missing or odd row, never a crash. The body, contact and joint lists are
// then walked from there, and the world's list heads found by value. Field offsets are in Layout; every row keeps its
// object's raw bytes too, so they can be rechecked offline (scripts/check_native.py). Output is derived from game data:
// never commit it.
static partial class NativeBox2D
{
    // Offsets in Unity 2020.3's fork, x64, read off the dumps (v2.3.1's in comments where they differ). Unity keeps
    // v2.3.1's structs and field order but inserts fields of its own: per-fixture mass data, two floats and an int in
    // b2Body, the island indices and four floats in b2Joint, extra words in b2Contact, dynamic arrays and a moved
    // m_flags in b2World. The revolute joint has v2.4's solver state (motor, lower and upper impulses, a 2x2 point
    // mass, the joint angle, the axial mass, no limit state) after v2.3.1's 3-vector impulse; prismatic and weld are
    // v2.3.1's. The world keeps no contact count.
    static class Layout
    {
        public const int BodyType = 0x00, BodyFlags = 0x04, BodyIsland = 0x08, BodyXf = 0x0c, BodySweep = 0x1c,
            BodyV = 0x40, BodyW = 0x48, BodyForce = 0x4c, BodyTorque = 0x54, BodyWorld = 0x58, BodyPrev = 0x60,
            BodyNext = 0x68, BodyFixtures = 0x70, BodyFixtureCount = 0x78, BodyJoints = 0x80, BodyContacts = 0x88,
            BodyMass = 0x90, BodyInvMass = 0x94, BodyUnity98 = 0x98, BodyI = 0xa0 /* 0x98 */, BodyInvI = 0xa4,
            BodyLinearDamping = 0xa8, BodyAngularDamping = 0xac, BodyGravityScale = 0xb0, BodySleepTime = 0xb4,
            BodyUnityB8 = 0xb8, BodyUserData = 0xc0 /* 0xb0 */, BodySize = 0xc8; // the next body block is 0xe0 on
        // b2Fixture: Unity puts b2MassData {mass, center, I} and one more float in front: v2.3.1 offsets + 0x10.
        public const int FixMassData = 0x00, FixUnity10 = 0x10, FixDensity = 0x14 /* 0x00 */, FixNext = 0x18, FixBody = 0x20,
            FixShape = 0x28, FixFriction = 0x30, FixRestitution = 0x34, FixProxies = 0x38, FixProxyCount = 0x40,
            FixFilter = 0x44, FixSensor = 0x4a, FixUserData = 0x50, FixSize = 0x58;
        public const int ShapeType = 0x08, ShapeRadius = 0x0c, CircleP = 0x10, PolyCentroid = 0x10, PolyVerts = 0x18,
            PolyNormals = 0x58, PolyCount = 0x98, PolySize = 0xa0;
        // b2Contact: Unity adds two ints before the manifold, the two shapes' radii after it, and 0x48 bytes more
        // before toiCount.
        public const int ContactFlags = 0x08, ContactPrev = 0x10, ContactNext = 0x18, ContactNodeA = 0x20, ContactNodeB = 0x40,
            ContactFixA = 0x60, ContactFixB = 0x68, ContactIndexA = 0x70, ContactIndexB = 0x74, ContactUnity78 = 0x78,
            ContactManifold = 0x80 /* 0x78 */, ContactRadii = 0xc0, ContactUnityC8 = 0xc8, ContactToiCount = 0x110 /* 0xb8 */,
            ContactToi = 0x114, ContactFriction = 0x118, ContactRestitution = 0x11c, ContactTangentSpeed = 0x120,
            ContactUnity124 = 0x124, ContactSize = 0x130;
        public const int EdgeOther = 0x00, EdgeTarget = 0x08, EdgeNext = 0x18; // b2ContactEdge and b2JointEdge
        public const int JointType = 0x08, JointPrev = 0x10, JointNext = 0x18, JointBodyA = 0x60, JointBodyB = 0x68,
            JointIndexA = 0x70, JointIndexB = 0x74, JointUnity78 = 0x78, JointUserData = 0x90 /* 0x78 */, JointBase = 0x98 /* 0x80 */;
        // b2World fields relative to m_bodyList, whose offset is found per world (0x19380 here, v2.3.1 0x19320).
        public const int WorldJointList = 0x08, WorldBodyCount = 0x10, WorldJointCount = 0x14, WorldGravity = 0x58 /* 0x18 */,
            WorldAllowSleep = 0x60, WorldInvDt0 = 0x78, WorldWarmStarting = 0x7c, WorldContinuous = 0x7d,
            WorldSubStepping = 0x7e, WorldStepComplete = 0x7f, WorldFlags = 0x80 /* -0x88 */;
    }

    // ---- guarded memory access ----

    [StructLayout(LayoutKind.Sequential)]
    struct MemoryInfo { public IntPtr Base, AllocationBase; public uint AllocationProtect; public IntPtr Size; public uint State, Protect, Type; }

    [DllImport("kernel32.dll")] static extern UIntPtr VirtualQuery(IntPtr address, out MemoryInfo info, UIntPtr length);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr GetModuleHandle(string name);

    const uint Commit = 0x1000, AnyRead = 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80, AnyWrite = 0x04 | 0x08 | 0x40 | 0x80,
        NoAccessOrGuard = 0x01 | 0x100;
    static readonly Dictionary<long, bool> pages = new Dictionary<long, bool>(); // per dump: memory moves between dumps

    static bool Query(long at, out MemoryInfo mi) =>
        VirtualQuery((IntPtr)at, out mi, (UIntPtr)Marshal.SizeOf(typeof(MemoryInfo))) != UIntPtr.Zero;

    static bool CanRead(long at, int n)
    {
        if (at < 0x10000 || at > 0x7FFFFFFF0000 || n <= 0) return false;
        for (long p = at & ~0xFFFL; p < at + n; p += 0x1000)
        {
            if (!pages.TryGetValue(p, out var ok))
                pages[p] = ok = Query(p, out var mi) && mi.State == Commit && (mi.Protect & AnyRead) != 0
                    && (mi.Protect & NoAccessOrGuard) == 0;
            if (!ok) return false;
        }
        return true;
    }

    static bool IsPtr(long q) => (q & 7) == 0 && q >= 0x10000 && q < 0x7FFFFFFF0000;

    // A copy of n native bytes at an address, or null if any of them is not readable.
    sealed class Blk
    {
        public readonly long At;
        public readonly byte[] B;
        Blk(long at, byte[] b) { At = at; B = b; }
        public static Blk Read(long at, int n)
        {
            if (!CanRead(at, n)) return null;
            var b = new byte[n];
            Marshal.Copy((IntPtr)at, b, 0, n);
            return new Blk(at, b);
        }
        public int I(int o) => BitConverter.ToInt32(B, o);
        public uint U(int o) => BitConverter.ToUInt32(B, o);
        public ushort U16(int o) => BitConverter.ToUInt16(B, o);
        public short I16(int o) => BitConverter.ToInt16(B, o);
        public float F(int o) => BitConverter.ToSingle(B, o);
        public long P(int o) => BitConverter.ToInt64(B, o);
        public Vector2 V(int o) => new Vector2(F(o), F(o + 4));
        public IEnumerable<float> Fs(int o, int n) => Enumerable.Range(0, n).Select(i => F(o + 4 * i));
        public string Hex(int n) => BitConverter.ToString(B, 0, Math.Min(n, B.Length)).Replace("-", "").ToLowerInvariant();
    }

    static string H(long a) => a == 0 ? null : "0x" + a.ToString("x");
    static uint Bits(float f) => BitConverter.ToUInt32(BitConverter.GetBytes(f), 0);

    static readonly FieldInfo cachedPtr = typeof(UnityEngine.Object).GetField("m_CachedPtr", BindingFlags.NonPublic | BindingFlags.Instance);
    static long Native(UnityEngine.Object o) => o == null ? 0 : ((IntPtr)cachedPtr.GetValue(o)).ToInt64();

    // ---- discovery ----

    // The b2Body of a Rigidbody2D: breadth-first through pointers (depth <= 2) below its native object, the first block
    // holding the body's exact position followed by a unit (sin, cos), of the right b2BodyType, with a world pointer.
    static long FindBody(Rigidbody2D rb, out string chain)
    {
        uint px = Bits(rb.position.x), py = Bits(rb.position.y);
        int type = 2 - (int)rb.bodyType; // Unity Dynamic 0 / Kinematic 1 / Static 2; Box2D static 0 / kinematic 1 / dynamic 2
        var queue = new Queue<(long at, int depth, string chain)>();
        var seen = new HashSet<long>();
        queue.Enqueue((Native(rb), 0, ""));
        while (queue.Count > 0)
        {
            var (at, depth, ch) = queue.Dequeue();
            if (!seen.Add(at)) continue;
            var b = Blk.Read(at, 0x200) ?? Blk.Read(at, 0x40);
            if (b == null) continue;
            for (int o = 0; o + 16 <= b.B.Length; o += 4)
            {
                if (b.U(o) != px || b.U(o + 4) != py) continue;
                float s = b.F(o + 8), c = b.F(o + 12);
                if (Mathf.Abs(s * s + c * c - 1f) > 1e-4f) continue;
                long body = at + o - Layout.BodyXf;
                var bb = Blk.Read(body, Layout.BodySize);
                if (bb == null || bb.I(Layout.BodyType) != type || !IsPtr(bb.P(Layout.BodyWorld))
                    || !CanRead(bb.P(Layout.BodyWorld), 8)) continue;
                chain = ch + (o == Layout.BodyXf ? "" : $"{o - Layout.BodyXf:+0;-0}");
                return body;
            }
            if (depth < 2)
                for (int o = 0; o + 8 <= b.B.Length; o += 8)
                    if (IsPtr(b.P(o))) queue.Enqueue((b.P(o), depth + 1, ch + $"[0x{o:x}]"));
        }
        chain = null;
        return 0;
    }

    // Where a native block points at one of the given Unity objects: directly, or one pointer away.
    static T Owner<T>(Blk b, Dictionary<long, T> natives, int from, out string at) where T : class
    {
        for (int o = from; o + 8 <= b.B.Length; o += 8)
            if (natives.TryGetValue(b.P(o), out var t)) { at = $"0x{o:x}"; return t; }
        for (int o = from; o + 8 <= b.B.Length; o += 8)
        {
            var inner = IsPtr(b.P(o)) ? Blk.Read(b.P(o), 0x40) : null;
            if (inner == null) continue;
            for (int i = 0; i < 0x40; i += 8)
                if (natives.TryGetValue(inner.P(i), out var t)) { at = $"[0x{o:x}]+0x{i:x}"; return t; }
        }
        at = null;
        return null;
    }

    static List<long> Chain(long head, int next, int limit = 1 << 20)
    {
        var list = new List<long>();
        var seen = new HashSet<long>();
        for (long p = head; IsPtr(p) && list.Count < limit && seen.Add(p);)
        {
            var b = Blk.Read(p, next + 8);
            if (b == null) break;
            list.Add(p);
            p = b.P(next);
        }
        return list;
    }

    static long Head(long p, int prev)
    {
        for (int i = 0; i < 1 << 20; i++)
        {
            var b = Blk.Read(p, prev + 8);
            if (b == null || !IsPtr(b.P(prev))) return p;
            p = b.P(prev);
        }
        return p;
    }

    // ---- the dump ----

    // A dump asked for at the next physics step is written right after that step's Simulate, before scripts run
    // again: the plugin's once-per-action pose update re-creates the climber's joints (zero impulses) after the last.
    static string pending;
    static ManualLogSource pendingLog;
    public static void WriteAfterNextStep(string file, ManualLogSource log) { pending = file; pendingLog = log; }
    public static void AfterSimulate()
    {
        if (pending == null) return;
        var file = pending;
        pending = null;
        Write(file, pendingLog);
    }

    // One row per world, body, fixture (with its shape), Unity collider (with GetContacts for the cross-check), contact
    // (in the world's contact-list order) and joint. scanSettings also searches memory for where the engine settings
    // live (it changes them for a moment, so not mid-run unless asked).
    public static void Write(string file, ManualLogSource log, bool scanSettings = false)
    {
        var watch = Stopwatch.StartNew();
        pages.Clear();
        var sb = new StringBuilder();
        var rbs = EngineDump.All<Rigidbody2D>(default).ToList();
        var cols = EngineDump.All<Collider2D>(default).ToList();
        var joints = EngineDump.All<Joint2D>(default).ToList();
        var rbByNative = rbs.Where(r => Native(r) != 0).ToDictionary(Native);
        var colByNative = cols.Where(c => Native(c) != 0).ToDictionary(Native);
        var jointByNative = joints.Where(j => Native(j) != 0).ToDictionary(Native);
        long unity = GetModuleHandle("UnityPlayer.dll").ToInt64();

        var bodyOf = new Dictionary<long, Rigidbody2D>();
        var worlds = new Dictionary<long, long>(); // world -> some body in it
        var found = new Json(sb, "discovery").F("unityPlayer", H(unity)).F("rigidbodies", rbs.Count);
        var chains = new List<string>();
        foreach (var rb in rbs.Where(r => r.gameObject.activeInHierarchy && r.simulated))
        {
            long body = FindBody(rb, out var chain);
            chains.Add($"{EngineDump.PathOf(rb.transform)} {(body == 0 ? "NOT FOUND" : chain)}");
            if (body == 0) continue;
            bodyOf[body] = rb;
            worlds[Blk.Read(body, Layout.BodySize).P(Layout.BodyWorld)] = body;
        }
        found.F("chains", chains).F("worlds", worlds.Keys.Select(H)).End();

        var fixtureIds = new Dictionary<long, (string path, int piece)>();
        foreach (var w in worlds)
            DumpWorld(sb, w.Key, w.Value, unity, bodyOf, rbByNative, colByNative, jointByNative, fixtureIds);

        // Unity's side of every collider, for checking the pieces and contacts above.
        var points = new ContactPoint2D[64];
        foreach (var c in cols)
        {
            var o = new Json(sb, "ucollider").F("path", EngineDump.PathOf(c.transform)).F("component", EngineDump.ComponentIndex(c))
                .F("native", H(Native(c)))
                .F("type", c.GetType().Name).F("active", c.enabled && c.gameObject.activeInHierarchy).F("trigger", c.isTrigger)
                .F("body", c.attachedRigidbody == null ? null : EngineDump.PathOf(c.attachedRigidbody.transform))
                .F("shapeCount", c.shapeCount).F("material", EngineDump.Mat(c.sharedMaterial))
                .F("friction", c.friction).F("bounciness", c.bounciness);
            int n = c.enabled && c.gameObject.activeInHierarchy ? c.GetContacts(points) : 0;
            var cp = new StringBuilder("[");
            for (int i = 0; i < n; i++)
            {
                var p = points[i];
                var e = new StringBuilder();
                new Json(e, "point").F("collider", EngineDump.PathOf(p.collider.transform))
                    .F("other", EngineDump.PathOf(p.otherCollider.transform)).F("point", p.point).F("normal", p.normal)
                    .F("separation", p.separation).F("normalImpulse", p.normalImpulse).F("tangentImpulse", p.tangentImpulse)
                    .F("relativeVelocity", p.relativeVelocity).F("enabled", p.enabled);
                cp.Append(i == 0 ? "" : ",").Append(e).Append('}');
            }
            o.F("contactCount", n).Raw("contacts", cp.Append(']').ToString()).End();
        }

        if (scanSettings) ScanSettings(sb, unity, worlds.Keys);
        File.WriteAllText(file, sb.ToString());
        log.LogInfo($"native Box2D dump written in {watch.ElapsedMilliseconds} ms: {file}");
    }

    static void DumpWorld(StringBuilder sb, long world, long anyBody, long unity, Dictionary<long, Rigidbody2D> bodyOf,
        Dictionary<long, Rigidbody2D> rbByNative, Dictionary<long, Collider2D> colByNative,
        Dictionary<long, Joint2D> jointByNative, Dictionary<long, (string path, int piece)> fixtureIds)
    {
        long head = Head(anyBody, Layout.BodyPrev);
        int size = 0x1A000; // v2.3.1's b2World is 0x19380 bytes, most of it the block allocator
        while (size > 0x1000 && !CanRead(world, size)) size -= 0x1000;
        var wb = Blk.Read(world, size);
        var hits = new List<int>();
        for (int o = 0; wb != null && o + 8 <= size; o += 8)
            if (wb.P(o) == head) hits.Add(o);
        int bl = hits.Count > 0 ? hits[hits.Count - 1] : -1;
        var w = new Json(sb, "world").F("addr", H(world)).F("bodyListOffset", bl < 0 ? null : $"0x{bl:x}")
            .F("bodyListHits", hits.Select(h => $"0x{h:x}"));
        var bodies = Chain(head, Layout.BodyNext);
        // The contact list is found the same way: the head reached from any body's contact edges, wherever the world
        // holds it (0x60 before the body list here).
        long contactHead = 0;
        foreach (long body in bodies)
        {
            var edge = Blk.Read(Blk.Read(body, Layout.BodySize)?.P(Layout.BodyContacts) ?? 0, 0x20);
            if (edge != null && IsPtr(edge.P(Layout.EdgeTarget))) { contactHead = Head(edge.P(Layout.EdgeTarget), Layout.ContactPrev); break; }
        }
        int cl = -1;
        for (int o = Math.Max(0, bl - 0x400); contactHead != 0 && o < bl; o += 8)
            if (wb.P(o) == contactHead) cl = o;
        var contacts = cl < 0 ? new List<long>() : Chain(contactHead, Layout.ContactNext);
        if (bl >= 0x100 && bl + 0x100 <= size)
        {
            w.F("contactListOffset", cl < 0 ? null : $"0x{cl:x}").F("contactsInList", contacts.Count).F("bodyCount", wb.I(bl + Layout.WorldBodyCount))
                .F("jointCount", wb.I(bl + Layout.WorldJointCount)).F("bodiesInList", bodies.Count)
                .F("gravity", wb.V(bl + Layout.WorldGravity)).F("allowSleep", wb.B[bl + Layout.WorldAllowSleep])
                .F("inv_dt0", wb.F(bl + Layout.WorldInvDt0)).F("warmStarting", wb.B[bl + Layout.WorldWarmStarting])
                .F("continuousPhysics", wb.B[bl + Layout.WorldContinuous]).F("subStepping", wb.B[bl + Layout.WorldSubStepping])
                .F("stepComplete", wb.B[bl + Layout.WorldStepComplete]).F("flags", wb.I(bl + Layout.WorldFlags))
                .F("rawFromBodyListMinus0x100", Blk.Read(world + bl - 0x100, 0x200)?.Hex(0x200));
        }
        w.End();

        // Bodies in list order: Box2D prepends, so this is newest first; island DFS seeds walk it in this order.
        for (int i = 0; i < bodies.Count; i++)
        {
            var b = Blk.Read(bodies[i], 0xe0) ?? Blk.Read(bodies[i], Layout.BodySize);
            bodyOf.TryGetValue(bodies[i], out var rb);
            var owner = Owner(b, rbByNative, Layout.BodyUserData, out var ownerAt);
            var o = new Json(sb, "body").F("index", i).F("addr", H(bodies[i])).F("world", H(world))
                .F("path", rb == null ? null : EngineDump.PathOf(rb.transform)).F("userDataRb", ownerAt)
                .F("userDataMatches", owner == rb)
                .F("type", b.I(Layout.BodyType)).F("flags", b.U16(Layout.BodyFlags)).F("islandIndex", b.I(Layout.BodyIsland))
                .F("xf", b.Fs(Layout.BodyXf, 4)).F("sweep", b.Fs(Layout.BodySweep, 9))
                .F("linearVelocity", b.V(Layout.BodyV)).F("angularVelocity", b.F(Layout.BodyW))
                .F("force", b.V(Layout.BodyForce)).F("torque", b.F(Layout.BodyTorque))
                .F("fixtureCount", b.I(Layout.BodyFixtureCount)).F("mass", b.F(Layout.BodyMass)).F("invMass", b.F(Layout.BodyInvMass))
                .F("I", b.F(Layout.BodyI)).F("invI", b.F(Layout.BodyInvI)).F("linearDamping", b.F(Layout.BodyLinearDamping))
                .F("angularDamping", b.F(Layout.BodyAngularDamping)).F("gravityScale", b.F(Layout.BodyGravityScale))
                .F("sleepTime", b.F(Layout.BodySleepTime)).F("unity98", b.Fs(Layout.BodyUnity98, 2)).F("unityB8", b.I(Layout.BodyUnityB8))
                .F("raw", b.Hex(b.B.Length));
            if (rb != null)
                o.F("apiPosition", rb.position).F("apiRotation", rb.rotation).F("apiVelocity", rb.velocity)
                    .F("apiAngularVelocity", rb.angularVelocity).F("apiMass", rb.mass).F("apiInertia", rb.inertia)
                    .F("apiCenterOfMass", rb.centerOfMass).F("apiWorldCenterOfMass", rb.worldCenterOfMass)
                    .F("apiAwake", rb.IsAwake()).F("apiUseAutoMass", rb.useAutoMass).Tf("tf", rb.transform);
            o.End();

            // Fixtures in the body's list order (also newest first). A Unity collider owns one fixture per piece.
            var perCollider = new Dictionary<Collider2D, int>();
            int k = 0;
            foreach (long f in Chain(b.P(Layout.BodyFixtures), Layout.FixNext))
                DumpFixture(sb, f, bodies[i], k++, colByNative, perCollider, fixtureIds);
        }

        for (int i = 0; i < contacts.Count; i++) DumpContact(sb, contacts[i], i, unity, fixtureIds);

        int ji = 0;
        if (bl >= 0)
            foreach (long j in Chain(wb.P(bl + Layout.WorldJointList), Layout.JointNext))
                DumpJoint(sb, j, ji++, unity, bodyOf, jointByNative);
    }

    static void DumpFixture(StringBuilder sb, long f, long body, int k, Dictionary<long, Collider2D> colByNative,
        Dictionary<Collider2D, int> perCollider, Dictionary<long, (string path, int piece)> fixtureIds)
    {
        var b = Blk.Read(f, 0x60) ?? Blk.Read(f, Layout.FixSize);
        var col = Owner(b, colByNative, Layout.FixUserData, out var ownerAt);
        int piece = -1;
        if (col != null)
        {
            perCollider.TryGetValue(col, out piece);
            perCollider[col] = piece + 1;
        }
        string path = col == null ? null : EngineDump.PathOf(col.transform);
        fixtureIds[f] = (path, piece);
        var o = new Json(sb, "fixture").F("addr", H(f)).F("body", H(body)).F("bodyListIndex", k)
            .F("collider", path).F("component", col == null ? -1 : EngineDump.ComponentIndex(col)).F("colliderNative", H(Native(col)))
            .F("piece", piece).F("userDataCollider", ownerAt)
            .F("massData", b.Fs(Layout.FixMassData, 4)).F("unity10", b.F(Layout.FixUnity10)).F("density", b.F(Layout.FixDensity)).F("friction", b.F(Layout.FixFriction)).F("restitution", b.F(Layout.FixRestitution))
            .F("categoryBits", b.U16(Layout.FixFilter)).F("maskBits", b.U16(Layout.FixFilter + 2))
            .F("groupIndex", b.I16(Layout.FixFilter + 4)).F("sensor", b.B[Layout.FixSensor]).F("proxyCount", b.I(Layout.FixProxyCount));
        if (col != null) o.F("colliderTrigger", col.isTrigger).F("colliderFriction", col.friction);
        int proxies = b.I(Layout.FixProxyCount);
        var px = proxies > 0 && proxies < 64 ? Blk.Read(b.P(Layout.FixProxies), 32 * proxies) : null;
        if (px != null) // b2FixtureProxy: tight AABB, fixture, childIndex, broad-phase proxyId
            o.F("proxyAabbs", Enumerable.Range(0, proxies).SelectMany(i => px.Fs(32 * i, 4)))
                .F("proxyIds", Enumerable.Range(0, proxies).Select(i => px.I(32 * i + 28)))
                .F("proxyChildren", Enumerable.Range(0, proxies).Select(i => px.I(32 * i + 24)));
        var s = Blk.Read(b.P(Layout.FixShape), Layout.PolySize + 8) ?? Blk.Read(b.P(Layout.FixShape), 0x18);
        if (s != null)
        {
            int type = s.I(Layout.ShapeType);
            o.F("shapeType", type).F("radius", s.F(Layout.ShapeRadius));
            if (type == 0) o.F("p", s.V(Layout.CircleP));
            if (type == 2 && s.B.Length >= Layout.PolySize)
            {
                int n = Mathf.Clamp(s.I(Layout.PolyCount), 0, 8);
                o.F("count", s.I(Layout.PolyCount)).F("centroid", s.V(Layout.PolyCentroid))
                    .F("vertices", s.Fs(Layout.PolyVerts, 2 * n)).F("normals", s.Fs(Layout.PolyNormals, 2 * n));
            }
            o.F("shapeRaw", s.Hex(s.B.Length));
        }
        o.F("raw", b.Hex(b.B.Length)).End();
    }

    static void DumpContact(StringBuilder sb, long c, int index, long unity, Dictionary<long, (string path, int piece)> fixtureIds)
    {
        var b = Blk.Read(c, 0x140) ?? Blk.Read(c, Layout.ContactSize);
        if (b == null) return;
        (string path, int piece) Id(long f) => fixtureIds.TryGetValue(f, out var id) ? id : (null, -1);
        var a = Id(b.P(Layout.ContactFixA));
        var bb = Id(b.P(Layout.ContactFixB));
        int m = Layout.ContactManifold, n = Mathf.Clamp(b.I(m + 0x3c), 0, 2);
        var pts = new StringBuilder("[");
        for (int i = 0; i < n; i++) // b2ManifoldPoint: localPoint, normalImpulse, tangentImpulse, b2ContactID
        {
            int p = m + 0x14 * i;
            var e = new StringBuilder();
            new Json(e, "mp").F("localPoint", b.V(p)).F("normalImpulse", b.F(p + 8)).F("tangentImpulse", b.F(p + 12))
                .F("key", b.U(p + 16)).F("cf", new[] { (int)b.B[p + 16], b.B[p + 17], b.B[p + 18], b.B[p + 19] });
            pts.Append(i == 0 ? "" : ",").Append(e).Append('}');
        }
        new Json(sb, "contact").F("index", index).F("addr", H(c)).F("vtable", b.P(0) - unity).F("flags", b.U(Layout.ContactFlags))
            .F("fixtureA", H(b.P(Layout.ContactFixA))).F("colliderA", a.path).F("pieceA", a.piece).F("childA", b.I(Layout.ContactIndexA))
            .F("fixtureB", H(b.P(Layout.ContactFixB))).F("colliderB", bb.path).F("pieceB", bb.piece).F("childB", b.I(Layout.ContactIndexB))
            .F("manifoldType", b.I(m + 0x38)).F("pointCount", b.I(m + 0x3c)).F("localNormal", b.V(m + 0x28))
            .F("localPoint", b.V(m + 0x30)).Raw("points", pts.Append(']').ToString())
            .F("unity78", new[] { b.I(Layout.ContactUnity78), b.I(Layout.ContactUnity78 + 4) }).F("radii", b.V(Layout.ContactRadii))
            .F("unityC8", b.Fs(Layout.ContactUnityC8, 18)).F("unity124", new[] { b.I(Layout.ContactUnity124), b.I(Layout.ContactUnity124 + 4) })
            .F("toiCount", b.I(Layout.ContactToiCount)).F("toi", b.F(Layout.ContactToi)).F("friction", b.F(Layout.ContactFriction))
            .F("restitution", b.F(Layout.ContactRestitution)).F("tangentSpeed", b.F(Layout.ContactTangentSpeed))
            .F("raw", b.Hex(b.B.Length)).End();
    }

    static void DumpJoint(StringBuilder sb, long j, int index, long unity, Dictionary<long, Rigidbody2D> bodyOf,
        Dictionary<long, Joint2D> jointByNative)
    {
        var b = Blk.Read(j, 0x150) ?? Blk.Read(j, 0x120) ?? Blk.Read(j, Layout.JointBase);
        if (b == null) return;
        string Body(long p) => bodyOf.TryGetValue(p, out var rb) ? EngineDump.PathOf(rb.transform) : null;
        var uj = Owner(b, jointByNative, 0x20, out var ownerAt);
        int type = b.I(Layout.JointType);
        var o = new Json(sb, "joint").F("index", index).F("addr", H(j)).F("vtable", b.P(0) - unity).F("type", type)
            .F("bodyA", Body(b.P(Layout.JointBodyA))).F("bodyB", Body(b.P(Layout.JointBodyB)))
            .F("indexA", b.I(Layout.JointIndexA)).F("indexB", b.I(Layout.JointIndexB)).F("unity78", b.Fs(Layout.JointUnity78, 4))
            .F("bytes0c", new[] { (int)b.B[0x0c], b.B[0x0d], b.B[0x0e], b.B[0x0f] }).F("userDataJoint", ownerAt)
            .F("joint", uj == null ? null : EngineDump.PathOf(uj.transform)).F("unityType", uj?.GetType().Name);
        const int x = Layout.JointBase;
        // Solver temporaries (r, local centres, inverse masses, effective masses) are left over from the last solve, or
        // from the block's previous owner for a joint created since.
        // Revolute: impulse.z stays 0 (GetReactionTorque still reads it, so Unity's reactionTorque is 0 for hinges); the
        // pot hinge at its -15 degree limit puts its push in lowerImpulse.
        if (type == 1 && b.B.Length >= x + 0x8c)
            o.F("localAnchorA", b.V(x)).F("localAnchorB", b.V(x + 8)).F("impulse", b.Fs(x + 0x10, 3)).F("motorImpulse", b.F(x + 0x1c))
                .F("lowerImpulse", b.F(x + 0x20)).F("upperImpulse", b.F(x + 0x24))
                .F("enableMotor", b.B[x + 0x28]).F("maxMotorTorque", b.F(x + 0x2c)).F("motorSpeed", b.F(x + 0x30))
                .F("enableLimit", b.B[x + 0x34]).F("referenceAngle", b.F(x + 0x38)).F("lowerAngle", b.F(x + 0x3c))
                .F("upperAngle", b.F(x + 0x40)).F("rA", b.V(x + 0x44)).F("rB", b.V(x + 0x4c)).F("localCenterA", b.V(x + 0x54))
                .F("localCenterB", b.V(x + 0x5c)).F("invMass", b.Fs(x + 0x64, 4)).F("K", b.Fs(x + 0x74, 4))
                .F("angle", b.F(x + 0x84)).F("axialMass", b.F(x + 0x88));
        if (type == 2 && b.B.Length >= x + 0xb4) // b2PrismaticJoint, v2.3.1
            o.F("localAnchorA", b.V(x)).F("localAnchorB", b.V(x + 8)).F("localXAxisA", b.V(x + 0x10)).F("localYAxisA", b.V(x + 0x18))
                .F("referenceAngle", b.F(x + 0x20)).F("impulse", b.Fs(x + 0x24, 3)).F("motorImpulse", b.F(x + 0x30))
                .F("lowerTranslation", b.F(x + 0x34)).F("upperTranslation", b.F(x + 0x38)).F("maxMotorForce", b.F(x + 0x3c))
                .F("motorSpeed", b.F(x + 0x40)).F("enableLimit", b.B[x + 0x44]).F("enableMotor", b.B[x + 0x45])
                .F("limitState", b.I(x + 0x48)).F("localCenterA", b.V(x + 0x4c)).F("localCenterB", b.V(x + 0x54))
                .F("invMass", b.Fs(x + 0x5c, 4)).F("axis", b.V(x + 0x6c)).F("perp", b.V(x + 0x74)).F("s1s2a1a2", b.Fs(x + 0x7c, 4))
                .F("K", b.Fs(x + 0x8c, 9)).F("motorMass", b.F(x + 0xb0));
        if (type == 8 && b.B.Length >= x + 0x84) // b2WeldJoint, v2.3.1
            o.F("frequencyHz", b.F(x)).F("dampingRatio", b.F(x + 4)).F("bias", b.F(x + 8)).F("localAnchorA", b.V(x + 0xc))
                .F("localAnchorB", b.V(x + 0x14)).F("referenceAngle", b.F(x + 0x1c)).F("gamma", b.F(x + 0x20))
                .F("impulse", b.Fs(x + 0x24, 3)).F("rA", b.V(x + 0x30)).F("rB", b.V(x + 0x38)).F("localCenterA", b.V(x + 0x40))
                .F("localCenterB", b.V(x + 0x48)).F("invMass", b.Fs(x + 0x50, 4)).F("mass", b.Fs(x + 0x60, 9));
        if (uj != null)
        {
            o.F("unityEnableCollision", uj.enableCollision).Tf("tf", uj.transform)
                .F("connected", uj.connectedBody == null ? null : EngineDump.PathOf(uj.connectedBody.transform))
                .Tf("connectedTf", uj.connectedBody == null ? null : uj.connectedBody.transform)
                .F("reactionForce", uj.GetReactionForce(Time.fixedDeltaTime)).F("reactionTorque", uj.GetReactionTorque(Time.fixedDeltaTime));
            if (uj is AnchoredJoint2D a) o.F("unityAnchor", a.anchor).F("unityConnectedAnchor", a.connectedAnchor);
            if (uj is HingeJoint2D h) o.F("unityLimitState", h.limitState.ToString()).F("unityReferenceAngle", h.referenceAngle);
            if (uj is SliderJoint2D s) o.F("unityLimitState", s.limitState.ToString()).F("unityAngle", s.angle);
        }
        o.F("raw", b.Hex(b.B.Length)).End();
    }

    // ---- where the engine settings live ----

    // Each setting is set to a sentinel for a moment and memory searched for it in the forms Box2D could store it in
    // (as set, per step, in radians, squared). Hits in UnityPlayer.dll's data or next to a world say which form the
    // solver reads, e.g. whether maxTranslationSpeed is a per-step b2_maxTranslation (T1). The values as they are now
    // (left by the last step) are searched in UnityPlayer.dll's data the same way.
    static void ScanSettings(StringBuilder sb, long unity, IEnumerable<long> worlds)
    {
        float dt = Time.fixedDeltaTime;
        var settings = new (string name, Func<float> get, Action<float> set, float sentinel)[]
        {
            ("maxTranslationSpeed", () => Physics2D.maxTranslationSpeed, v => Physics2D.maxTranslationSpeed = v, 100.375f),
            ("maxRotationSpeed", () => Physics2D.maxRotationSpeed, v => Physics2D.maxRotationSpeed = v, 361.5f),
            ("baumgarteScale", () => Physics2D.baumgarteScale, v => Physics2D.baumgarteScale = v, 0.40625f),
            ("baumgarteTOIScale", () => Physics2D.baumgarteTOIScale, v => Physics2D.baumgarteTOIScale = v, 0.765625f),
            ("maxLinearCorrection", () => Physics2D.maxLinearCorrection, v => Physics2D.maxLinearCorrection = v, 0.203125f),
            ("maxAngularCorrection", () => Physics2D.maxAngularCorrection, v => Physics2D.maxAngularCorrection = v, 8.125f),
            ("linearSleepTolerance", () => Physics2D.linearSleepTolerance, v => Physics2D.linearSleepTolerance = v, 0.0107421875f),
            ("angularSleepTolerance", () => Physics2D.angularSleepTolerance, v => Physics2D.angularSleepTolerance = v, 2.125f),
            ("timeToSleep", () => Physics2D.timeToSleep, v => Physics2D.timeToSleep = v, 0.53125f),
            ("velocityThreshold", () => Physics2D.velocityThreshold, v => Physics2D.velocityThreshold = v, 1.0625f),
        };
        var forms = new (string name, Func<float, float> f)[]
        {
            ("value", v => v), ("perStep", v => v * dt), ("perSecond", v => v / dt), ("rad", v => v * Mathf.Deg2Rad),
            ("radPerStep", v => v * Mathf.Deg2Rad * dt), ("squared", v => v * v), ("perStepSquared", v => v * dt * (v * dt)),
            ("radSquared", v => v * Mathf.Deg2Rad * (v * Mathf.Deg2Rad)),
            ("radPerStepSquared", v => v * Mathf.Deg2Rad * dt * (v * Mathf.Deg2Rad * dt)),
        };
        var now = new Dictionary<uint, string>();
        var sentinel = new Dictionary<uint, string>();
        foreach (var s in settings)
            foreach (var f in forms)
            {
                now[Bits(f.f(s.get()))] = $"{s.name}:{f.name}";
                sentinel[Bits(f.f(s.sentinel))] = $"{s.name}:{f.name}";
            }
        now[Bits(0.005f)] = "linearSlop?:value";
        now[Bits(Physics2D.defaultContactOffset)] = "defaultContactOffset:value";

        long image = 0;
        var nt = Blk.Read(unity, 0x1000);
        if (nt != null) image = BitConverter.ToUInt32(nt.B, nt.I(0x3c) + 0x50); // PE32+ OptionalHeader.SizeOfImage
        bool InImage(long a) => a >= unity && a < unity + image;
        var nearWorld = worlds.ToList();
        string Where(long a)
        {
            if (InImage(a)) return $"UnityPlayer+0x{a - unity:x}";
            foreach (var w in nearWorld) if (a >= w && a < w + 0x1A000) return $"world {H(w)}+0x{a - w:x}";
            return H(a);
        }

        var old = settings.Select(s => s.get()).ToArray();
        var rows = new List<string>();
        try
        {
            foreach (var s in settings) s.set(s.sentinel);
            Scan(sentinel, a => true, (a, what) => rows.Add($"sentinel {what} at {Where(a)}"), rows);
        }
        finally
        {
            for (int i = 0; i < settings.Length; i++) settings[i].set(old[i]);
        }
        Scan(now, InImage, (a, what) => rows.Add($"current {what} at {Where(a)}"), rows);
        new Json(sb, "settingsScan").F("dt", dt).F("unityPlayerSize", image).F("hits", rows).End();
    }

    // Every committed writable private or image region (<= 256 MB each, 4 GB in all), for 4-byte-aligned values,
    // with the 16 floats either side of each hit (capped at 400 hits).
    static unsafe void Scan(Dictionary<uint, string> values, Func<long, bool> keep, Action<long, string> hit, List<string> rows)
    {
        long total = 0;
        int hits = 0;
        var filter = new bool[1 << 16];
        foreach (var v in values.Keys) filter[v & 0xFFFF] = true;
        for (long a = 0x10000; a < 0x7FFFFFFF0000 && total < 4L << 30;)
        {
            if (!Query(a, out var mi)) break;
            long start = mi.Base.ToInt64(), size = mi.Size.ToInt64();
            a = start + size;
            if (mi.State != Commit || (mi.Protect & AnyWrite) == 0 || (mi.Protect & NoAccessOrGuard) != 0
                || (mi.Type != 0x20000 && mi.Type != 0x1000000) || size > 256L << 20) continue;
            total += size;
            uint* p = (uint*)start;
            for (long i = 0, n = size / 4; i < n; i++)
            {
                uint x = p[i];
                if (!filter[x & 0xFFFF] || !values.TryGetValue(x, out var what)) continue;
                long at = start + 4 * i;
                if (!keep(at) || ++hits > 400) continue;
                var ctx = Blk.Read(at - 64, 132);
                hit(at, ctx == null ? what : $"{what} context {string.Join(" ", ctx.Fs(0, 33).Select(Json.Num).ToArray())}");
            }
        }
        rows.Add($"scanned {total >> 20} MB, {hits} hits");
    }
}
