using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// The step recorder's view of the native Box2D world (see GameRecord.cs): the same structs NativeBox2D's JSON dump
// reads, written as compact little-endian binary for one physics step (before and after Simulate) in a few ms.
// Layouts and the Python decoder: goi_jax/parity/states.py. Output is derived from game data: never commit it.
static partial class NativeBox2D
{
    // Raw bytes kept per object. Body: the whole b2Body (pointers included, decoded offline). Joint: base + the largest
    // derived class (prismatic ends at 0x98 + 0xb4). Contact: three slices of b2Contact; the 0xc8..0x10f Unity words
    // and the list/edge pointers are left out.
    public const int RecBodyBytes = Layout.BodySize, RecJointBytes = 0x150, RecContactA = 0x08, RecContactALen = 4,
        RecContactB = Layout.ContactIndexA, RecContactBLen = Layout.ContactUnityC8 - Layout.ContactIndexA,
        RecContactC = Layout.ContactToiCount, RecContactCLen = Layout.ContactUnity124 - Layout.ContactToiCount,
        RecFixtureBytes = Layout.FixSize, RecShapeBytes = Layout.PolySize;

    // Lookup tables for one physics world; rebuilt when the world changes (every restore makes a fresh one).
    internal sealed class RecordMaps
    {
        public long World, BodyListHead;
        public int BodyListOffset = -1;
        public readonly Dictionary<long, int> Rb = new Dictionary<long, int>();       // native Rigidbody2D -> body index
        public readonly Dictionary<long, int> Col = new Dictionary<long, int>();      // native Collider2D -> collider index
        public readonly Dictionary<long, int> Joint = new Dictionary<long, int>();    // native Joint2D -> joint index
        public readonly Dictionary<long, (int col, int piece)> Fixture = new Dictionary<long, (int, int)>();
        public readonly List<long> Bodies = new List<long>(); // body list at build time, newest first (ground last)
    }

    // The b2Body of a Rigidbody2D: native Rigidbody2D + 0x78 points at it in this build (docs: native-box2d.md §2);
    // checked by the position bits, with the pose search as fallback.
    static long BodyOf(Rigidbody2D rb)
    {
        long n = Native(rb);
        var p = Blk.Read(n + 0x78, 8);
        if (p != null && IsPtr(p.P(0)))
        {
            var b = Blk.Read(p.P(0), Layout.BodySize);
            if (b != null && b.U(Layout.BodyXf) == Bits(rb.position.x) && b.U(Layout.BodyXf + 4) == Bits(rb.position.y))
                return p.P(0);
        }
        return FindBody(rb, out _);
    }

    // Builds the tables for the world `anchor` lives in, or returns the current ones if that world is unchanged.
    internal static RecordMaps Maps(RecordMaps maps, Rigidbody2D anchor, Rigidbody2D[] bodies, Collider2D[] colliders,
        Joint2D[] joints)
    {
        pages.Clear();
        long body = BodyOf(anchor);
        if (body == 0) return null;
        long world = Blk.Read(body, Layout.BodySize).P(Layout.BodyWorld);
        long head = Head(body, Layout.BodyPrev);
        if (maps != null && maps.World == world && maps.BodyListHead == head) return maps;
        var m = new RecordMaps { World = world, BodyListHead = head };
        for (int i = 0; i < bodies.Length; i++) if (Native(bodies[i]) != 0) m.Rb[Native(bodies[i])] = i;
        for (int i = 0; i < colliders.Length; i++) if (Native(colliders[i]) != 0) m.Col[Native(colliders[i])] = i;
        for (int i = 0; i < joints.Length; i++) if (Native(joints[i]) != 0) m.Joint[Native(joints[i])] = i;
        int size = 0x1A000;
        while (size > 0x1000 && !CanRead(world, size)) size -= 0x1000;
        var wb = Blk.Read(world, size);
        for (int o = 0; wb != null && o + 8 <= size; o += 8) if (wb.P(o) == head) m.BodyListOffset = o;
        m.Bodies.AddRange(Chain(head, Layout.BodyNext));
        foreach (long b in m.Bodies)
        {
            var bb = Blk.Read(b, Layout.BodySize);
            var perCollider = new Dictionary<int, int>();
            foreach (long f in Chain(bb.P(Layout.BodyFixtures), Layout.FixNext))
            {
                var fb = Blk.Read(f, Layout.FixSize);
                int col = fb != null && m.Col.TryGetValue(fb.P(Layout.FixUserData), out var c) ? c : -1;
                perCollider.TryGetValue(col, out int piece);
                perCollider[col] = piece + 1;
                m.Fixture[f] = (col, piece);
            }
        }
        return m;
    }

    static int RbIndex(RecordMaps m, Blk body)
    {
        long ud = body.P(Layout.BodyUserData);
        return m.Rb.TryGetValue(ud, out int i) ? i : -1;
    }

    // One snapshot: the world's fields after m_bodyList (gravity, inv_dt0, flags...), every non-static body in list
    // order, every joint in list order, every contact in list order.
    internal static void Snapshot(BinaryWriter w, RecordMaps m)
    {
        pages.Clear();
        var world = m.BodyListOffset >= 0 ? Blk.Read(m.World + m.BodyListOffset, 0x88) : null;
        w.Write(world != null ? 0x88 : 0);
        if (world != null) w.Write(world.B);

        long head = Head(m.BodyListHead, Layout.BodyPrev); // a body created since (the gift) would now be first
        var bodies = Chain(head, Layout.BodyNext);
        var blocks = new List<Blk>(bodies.Count);
        foreach (long b in bodies) blocks.Add(Blk.Read(b, Layout.BodySize));
        int dynamic = 0;
        foreach (var b in blocks) if (b != null && b.I(Layout.BodyType) != 0) dynamic++;
        w.Write(dynamic);
        long jointHead = 0, contactHead = 0;
        foreach (var b in blocks)
        {
            if (b == null) continue;
            if (jointHead == 0)
            {
                var je = Blk.Read(b.P(Layout.BodyJoints), 0x20);
                if (je != null && IsPtr(je.P(Layout.EdgeTarget))) jointHead = Head(je.P(Layout.EdgeTarget), Layout.JointPrev);
            }
            if (contactHead == 0)
            {
                var ce = Blk.Read(b.P(Layout.BodyContacts), 0x20);
                if (ce != null && IsPtr(ce.P(Layout.EdgeTarget))) contactHead = Head(ce.P(Layout.EdgeTarget), Layout.ContactPrev);
            }
            if (b.I(Layout.BodyType) == 0) continue;
            w.Write(RbIndex(m, b));
            w.Write(b.B, 0, RecBodyBytes);
        }

        var joints = jointHead == 0 ? new List<long>() : Chain(jointHead, Layout.JointNext);
        w.Write(joints.Count);
        foreach (long j in joints)
        {
            var b = Blk.Read(j, RecJointBytes) ?? Blk.Read(j, 0x120) ?? Blk.Read(j, Layout.JointBase);
            var owner = b == null ? 0 : b.P(Layout.JointUserData);
            w.Write(m.Joint.TryGetValue(owner, out int ji) ? ji : -1);
            var raw = new byte[RecJointBytes];
            if (b != null) Buffer.BlockCopy(b.B, 0, raw, 0, Math.Min(b.B.Length, RecJointBytes));
            w.Write(raw);
        }

        var contacts = contactHead == 0 ? new List<long>() : Chain(contactHead, Layout.ContactNext);
        w.Write(contacts.Count);
        foreach (long c in contacts)
        {
            var b = Blk.Read(c, Layout.ContactSize);
            if (b == null) { w.Write(new byte[16 + RecContactALen + RecContactBLen + RecContactCLen]); continue; }
            var fa = m.Fixture.TryGetValue(b.P(Layout.ContactFixA), out var a) ? a : (-1, -1);
            var fb = m.Fixture.TryGetValue(b.P(Layout.ContactFixB), out var bb) ? bb : (-1, -1);
            w.Write(fa.Item1); w.Write(fa.Item2); w.Write(fb.Item1); w.Write(fb.Item2);
            w.Write(b.B, RecContactA, RecContactALen);
            w.Write(b.B, RecContactB, RecContactBLen);
            w.Write(b.B, RecContactC, RecContactCLen);
        }
    }

    // Every fixture of every body in list order (the static ground body holds the level's pieces): the owning body's
    // index (-1: the ground), collider index, piece, the b2Fixture's and its shape's raw bytes.
    internal static void Geometry(BinaryWriter w, RecordMaps m)
    {
        pages.Clear();
        w.Write(m.Bodies.Count);
        foreach (long b in m.Bodies)
        {
            var bb = Blk.Read(b, Layout.BodySize);
            var fixtures = Chain(bb.P(Layout.BodyFixtures), Layout.FixNext);
            w.Write(RbIndex(m, bb));
            w.Write(bb.I(Layout.BodyType));
            w.Write(fixtures.Count);
            foreach (long f in fixtures)
            {
                var fb = Blk.Read(f, Layout.FixSize);
                var id = m.Fixture.TryGetValue(f, out var x) ? x : (-1, -1);
                w.Write(id.Item1); w.Write(id.Item2);
                var fraw = new byte[RecFixtureBytes];
                var sraw = new byte[RecShapeBytes];
                if (fb != null)
                {
                    Buffer.BlockCopy(fb.B, 0, fraw, 0, RecFixtureBytes);
                    var s = Blk.Read(fb.P(Layout.FixShape), RecShapeBytes) ?? Blk.Read(fb.P(Layout.FixShape), 0x18);
                    if (s != null) Buffer.BlockCopy(s.B, 0, sraw, 0, s.B.Length);
                }
                w.Write(fraw);
                w.Write(sraw);
            }
        }
    }
}
