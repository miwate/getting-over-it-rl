using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using BepInEx.Logging;
using HarmonyLib;
using UnityEngine;
using Object = UnityEngine.Object;

// The per-step recorder ('w' at K = 1, 'W' at the configured K) and the controller constants ('q'), for physics parity
// work (goi_jax/parity). RecordAction ('W') runs one whole action as Step does (K physics steps, then the pose update
// and joint re-creation once) and returns one record per physics step. Version 3 adds the climber's Transforms
// (TFM0..2 with SCR0..2, TFM3 after the pose update on an action's last step).
//
// Record(action) runs exactly what Step(action) runs with K = 1 (friction scripts, controller, crane, Simulate,
// callbacks, teleports, then the once-per-action pose update) and returns one binary record of that step:
//   script state before the scripts (SCR0), right after PlayerControl.FixedUpdate (SCR1: what the controller computed:
//   motor speeds it wrote, the cursor velocity it set, its new mouse state) and after the step (SCR2); the native
//   Box2D world just before Simulate (NAT0: bodies, joints, contacts with manifolds and impulses) and just after it
//   (NAT1, before the pose update re-creates the climber's joints); and the events in between (EVTS): every
//   Rigidbody2D.MovePosition call (the cursor's target), every HammerCollisions / PotSounds collision callback in the
//   order Unity sent them (with the slide flag after it, and the ContactPoint2Ds each one saw: CPTS), and
//   HammerReturn calls.
// Layout (little-endian): int32 magic 0x31524F47 ("GOR1"), int32 version, then sections {int32 tag, int32 bytes,
// payload}; the decoder is goi_jax/parity/states.py. Output is derived from game data: never commit it.
partial class Game
{
    public const int RecordVersion = 3;
    // Script-state floats per snapshot, in this order (bools 0/1, ints exact); then the hammer's and the pot's tracked
    // Terrain colliders (int32 count, int32 collider indices). Names are served by 'q' too.
    public static readonly string[] ScriptFields =
    {
        "mouseInput.x", "mouseInput.y", "oldMouse.x", "oldMouse.y", "mouseVelocityAverage", "mw.x", "mw.y", "mouseSnap",
        "pauseInputTimer", "inputsToSkip", "oldAngle",
        "hj.motorSpeed", "hj.maxMotorTorque", "sj.motorSpeed", "sj.maxMotorForce",
        "pc.motor.motorSpeed", "pc.motor.maxMotorTorque", "pc.slider.motorSpeed", "pc.slider.maxMotorTorque",
        "hj.jointAngle", "hj.referenceAngle", "sj.reactionTorque", "sj.jointTranslation",
        "P.x", "P.y", "P.z", "T.x", "T.y", "T.z", "H.x", "H.y", "H.z",          // Transform positions the controller reads
        "cursor.x", "cursor.y", "cursor.vx", "cursor.vy", "cursorTf.x", "cursorTf.y", "cursorTf.z",
        "tip.vx", "tip.vy", "pot.vx", "pot.vy",
        "hammerSlide", "potSlide", "hammerFriction", "potBottomFriction",
        "simTime", "gravity.x", "gravity.y", "sensitivity", "trackpad", "loadFinished",
        "pot.x", "pot.y", "tip.x", "tip.y", "hub.x", "hub.y", "pot.rotation", "tip.rotation", "hub.rotation", "arm.rotation",
        "sj.reactionForce.x", "sj.reactionForce.y", "hj.reactionForce.x", "hj.reactionForce.y",
    };
    public const int TagScript = 0x30524353, TagNative = 0x3054414E, TagEvents = 0x53545645, TagTime = 0x454D4954, // "SCR0", "NAT0", "EVTS", "TIME"
        TagPoints = 0x53545043, // "CPTS": per collision callback, the ContactPoint2Ds Unity reported (GetContacts)
        TagTransforms = 0x304D4654; // "TFM0": the climber's Transforms (RecTransforms)
    // Event kinds (EVTS rows: phase, kind, script, collider, otherCollider, count, a, b, c, d, flags).
    public const int EvEnter = 1, EvStay = 2, EvExit = 3, EvHammerReturn = 4, EvMovePosition = 5;
    // Phases: 0 scripts (friction + controller), 1 between them and Simulate (the crane), 2 inside Simulate (callbacks),
    // 3 after Simulate (teleports, pose update).
    public int RecordPhase;

    MemoryStream recBytes;
    BinaryWriter rec;
    MemoryStream recEvents;
    BinaryWriter recEv;
    int recEventCount;
    MemoryStream recPoints;
    BinaryWriter recPt;
    int recPointCount;
    readonly ContactPoint2D[] recContactBuffer = new ContactPoint2D[64];
    NativeBox2D.RecordMaps recMaps;
    Joint2D[] recJoints;
    Dictionary<Collider2D, int> recColliderIndex;
    readonly Stopwatch recWatch = new Stopwatch();
    long recNativeTicks;
    Collider2D hammerCollider;

    public static Game Recording { get; private set; } // the game whose step is being recorded (event hooks)

    Joint2D[] RecJoints => recJoints ?? (recJoints = InLevel<Joint2D>());

    int ColliderIndex(Collider2D c)
    {
        if (c == null) return -1;
        if (recColliderIndex == null)
        {
            recColliderIndex = new Dictionary<Collider2D, int>();
            for (int i = 0; i < colliders.Length; i++) recColliderIndex[colliders[i]] = i;
        }
        return recColliderIndex.TryGetValue(c, out int k) ? k : -1;
    }

    public byte[] Record(Vector2 action)
    {
        int k = K;
        K = 1;
        try
        {
            return RecordSteps(action)[0];
        }
        finally
        {
            K = k;
        }
    }

    // 'W': one action at the configured K, one record per physics step (Step's loop cuts them: RecNextStep). Only the
    // last record holds the pose update's events and the Transforms after it (TFM3).
    public byte[] RecordAction(Vector2 action)
    {
        var steps = RecordSteps(action);
        using (var m = new MemoryStream())
        using (var w = new BinaryWriter(m))
        {
            w.Write(steps.Count);
            foreach (var r in steps)
            {
                w.Write(r.Length);
                w.Write(r);
            }
            w.Flush();
            return m.ToArray();
        }
    }

    List<byte[]> recSteps;

    List<byte[]> RecordSteps(Vector2 action)
    {
        recSteps = new List<byte[]>();
        BeginRecord();
        Recording = this;
        try
        {
            Step(action);
            RecTransforms(3); // after the once-per-action pose update
            recSteps.Add(EndRecord(action));
            return recSteps;
        }
        finally
        {
            Recording = null;
            rec = null; recBytes = null; recEv = null; recEvents = null; recPt = null; recPoints = null;
            recSteps = null;
        }
    }

    // Hook at the top of every physics step after the first in Step(): closes the previous step's record.
    void RecNextStep()
    {
        if (rec == null || recSteps == null) return;
        recSteps.Add(EndRecord(VirtualMouse.Input));
        BeginRecord();
    }

    void BeginRecord()
    {
        recWatch.Restart();
        recNativeTicks = 0;
        recBytes = new MemoryStream(32 << 10);
        rec = new BinaryWriter(recBytes);
        recEvents = new MemoryStream(1 << 10);
        recEv = new BinaryWriter(recEvents);
        recEventCount = 0;
        recPoints = new MemoryStream(1 << 10);
        recPt = new BinaryWriter(recPoints);
        recPointCount = 0;
        rec.Write(0x31524F47);
        rec.Write(RecordVersion);
    }

    byte[] EndRecord(Vector2 action)
    {
        Section(TagEvents, w => { w.Write(recEventCount); w.Write(recEvents.ToArray()); });
        Section(TagPoints, w => { w.Write(recPointCount); w.Write(recPoints.ToArray()); });
        long total = recWatch.ElapsedTicks;
        Section(TagTime, w =>
        {
            w.Write((float)(total * 1e3 / Stopwatch.Frequency));        // ms for the whole recorded step
            w.Write((float)(recNativeTicks * 1e3 / Stopwatch.Frequency)); // ms of it reading native memory
            w.Write(generation);
            w.Write(VirtualMouse.Input.x); w.Write(VirtualMouse.Input.y); // the action the controller read
        });
        return recBytes.ToArray();
    }

    // TFM<which> (record version >= 3): the climber's Transforms, Player down to Tip, then PoseControl's three mesh
    // hubs (hammer, dude, pot): per Transform its path, then localPosition, localRotation, localScale, position, rotation,
    // lossyScale (20 floats, exact). Written with SCR0/1/2 and once more after the pose update (3).
    Transform[] recTransforms;

    void RecTransforms(int which)
    {
        if (rec == null) return;
        if (recTransforms == null)
        {
            var chain = new List<Transform>();
            for (var t = tip.transform; t != null; t = t.parent)
            {
                chain.Insert(0, t);
                if (t == player.transform) break;
            }
            chain.Add(pose.hammerMeshHub); chain.Add(pose.dudeMeshHub); chain.Add(pose.potMeshHub);
            recTransforms = chain.ToArray();
        }
        Section(TagTransforms + (which << 24), w =>
        {
            w.Write(recTransforms.Length);
            foreach (var t in recTransforms)
            {
                var name = Encoding.UTF8.GetBytes(t == null ? "" : EngineDump.PathOf(t));
                w.Write(name.Length);
                w.Write(name);
                void V3(Vector3 v) { w.Write(v.x); w.Write(v.y); w.Write(v.z); }
                void Q(Quaternion q) { w.Write(q.x); w.Write(q.y); w.Write(q.z); w.Write(q.w); }
                if (t == null)
                {
                    for (int i = 0; i < 20; i++) w.Write(float.NaN);
                    continue;
                }
                V3(t.localPosition); Q(t.localRotation); V3(t.localScale); V3(t.position); Q(t.rotation); V3(t.lossyScale);
            }
        });
    }

    void Section(int tag, Action<BinaryWriter> body)
    {
        using (var m = new MemoryStream())
        using (var w = new BinaryWriter(m))
        {
            body(w);
            w.Flush();
            rec.Write(tag);
            rec.Write((int)m.Length);
            rec.Write(m.GetBuffer(), 0, (int)m.Length);
        }
    }

    // Hooks in Step(); no-ops unless recording.
    void RecScripts(int which)
    {
        if (rec == null) return;
        Section(TagScript + (which << 24), WriteScriptState);
        RecTransforms(which);
    }

    void RecNative(int which)
    {
        if (rec == null) return;
        long t0 = recWatch.ElapsedTicks;
        recMaps = NativeBox2D.Maps(recMaps, pot, bodies, colliders, RecJoints);
        if (recMaps == null) Section(TagNative + (which << 24), w => w.Write(-1));
        else Section(TagNative + (which << 24), w => NativeBox2D.Snapshot(w, recMaps));
        recNativeTicks += recWatch.ElapsedTicks - t0;
    }

    void WriteScriptState(BinaryWriter w)
    {
        hammerCollider = hammerCollider ?? hammer.GetComponent<Collider2D>();
        var f = new float[ScriptFields.Length];
        int i = 0;
        void V2(Vector2 v) { f[i++] = v.x; f[i++] = v.y; }
        void V3(Vector3 v) { f[i++] = v.x; f[i++] = v.y; f[i++] = v.z; }
        V2((Vector2)F.MouseInput.GetValue(pc)); V2((Vector2)F.OldMouse.GetValue(pc));
        f[i++] = (float)F.MouseSpeed.GetValue(pc); V2((Vector2)F.Mw.GetValue(pc));
        f[i++] = (bool)F.MouseSnap.GetValue(pc) ? 1 : 0;
        f[i++] = (float)F.PauseInput.GetValue(pc); f[i++] = (int)F.InputsToSkip.GetValue(pc);
        f[i++] = (float)R.OldAngle.GetValue(pc);
        f[i++] = pc.hj.motor.motorSpeed; f[i++] = pc.hj.motor.maxMotorTorque;
        f[i++] = pc.sj.motor.motorSpeed; f[i++] = pc.sj.motor.maxMotorTorque;
        var cm = (JointMotor2D)R.Motor.GetValue(pc);
        var cs = (JointMotor2D)R.Slider.GetValue(pc);
        f[i++] = cm.motorSpeed; f[i++] = cm.maxMotorTorque; f[i++] = cs.motorSpeed; f[i++] = cs.maxMotorTorque;
        f[i++] = pc.hj.jointAngle; f[i++] = pc.hj.referenceAngle; f[i++] = pc.sj.reactionTorque; f[i++] = pc.sj.jointTranslation;
        V3(pc.transform.position); V3(pc.tip.position); V3(pc.hj.transform.position);
        V2(cursor.position); V2(cursor.velocity); V3(cursor.transform.position);
        V2(tip.velocity); V2(pot.velocity);
        f[i++] = (bool)F.HammerSlide.GetValue(hammer) ? 1 : 0;
        f[i++] = (bool)F.PotSlide.GetValue(potSounds) ? 1 : 0;
        f[i++] = hammerCollider.friction;
        f[i++] = potSounds.bottomCol != null ? potSounds.bottomCol.friction : float.NaN;
        f[i++] = simTime; V2(Physics2D.gravity);
        f[i++] = (float)F.Sensitivity.GetValue(pc); f[i++] = pc.trackpad ? 1 : 0; f[i++] = pc.loadFinished ? 1 : 0;
        V2(pot.position); V2(tip.position); V2(hub.position);
        f[i++] = pot.rotation; f[i++] = tip.rotation; f[i++] = hub.rotation; f[i++] = arm.rotation;
        V2(pc.sj.reactionForce); V2(pc.hj.reactionForce);
        w.Write(f.Length);
        foreach (var x in f) w.Write(x);
        foreach (var keys in new[] { Keys(F.HammerPoints.GetValue(hammer)), Keys(F.PotPoints.GetValue(potSounds)) })
        {
            w.Write(keys.Length);
            foreach (var c in keys) w.Write(ColliderIndex(c));
        }
    }

    // ---- event hooks (Harmony; installed in RL mode by RecordHooks.Install) ----

    void Event(int kind, int script, int collider, int other, int count, float a, float b, float c, float d, int flags)
    {
        if (recEv == null) return;
        recEventCount++;
        recEv.Write(RecordPhase); recEv.Write(kind); recEv.Write(script); recEv.Write(collider); recEv.Write(other);
        recEv.Write(count); recEv.Write(a); recEv.Write(b); recEv.Write(c); recEv.Write(d); recEv.Write(flags);
    }

    internal void OnCollision(object script, int kind, Collision2D coll)
    {
        bool isHammer = script is HammerCollisions;
        bool slide = (bool)(isHammer ? F.HammerSlide.GetValue(hammer) : F.PotSlide.GetValue(potSounds));
        var points = isHammer ? F.HammerPoints.GetValue(hammer) : F.PotPoints.GetValue(potSounds);
        int tracked = points == null ? 0 : ((Dictionary<Collider2D, ContactPoint2D[]>)points).Count;
        var rv = coll.relativeVelocity;
        Event(kind, isHammer ? 0 : 1, ColliderIndex(coll.collider), ColliderIndex(coll.otherCollider), coll.contactCount,
            rv.x, rv.y, coll.gameObject.layer, 0, (slide ? 1 : 0) | (tracked << 8));
        // The points this callback saw, as GetContacts returns them (CPTS rows: event index, point, normal,
        // separation, normalImpulse, tangentImpulse, then the collider indices of the point's own pair).
        int n = recPt == null ? 0 : coll.GetContacts(recContactBuffer);
        for (int i = 0; i < n; i++)
        {
            var p = recContactBuffer[i];
            recPointCount++;
            recPt.Write(recEventCount - 1);
            recPt.Write(p.point.x); recPt.Write(p.point.y); recPt.Write(p.normal.x); recPt.Write(p.normal.y);
            recPt.Write(p.separation); recPt.Write(p.normalImpulse); recPt.Write(p.tangentImpulse);
            recPt.Write(ColliderIndex(p.collider)); recPt.Write(ColliderIndex(p.otherCollider));
        }
    }

    internal void OnHammerReturn(bool after)
    {
        Event(EvHammerReturn, after ? 1 : 0, -1, -1, 0, pc.tip.position.y, pc.transform.position.y, 0, 0,
            (bool)F.MouseSnap.GetValue(pc) ? 1 : 0);
    }

    internal void OnMovePosition(Rigidbody2D body, Vector2 target)
    {
        Event(EvMovePosition, Array.IndexOf(bodies, body), -1, -1, 0, target.x, target.y, body.position.x, body.position.y, 0);
    }

    // ---- 'g': the world's fixtures (convex pieces) and shapes, once per session ----
    public byte[] Geometry()
    {
        recMaps = NativeBox2D.Maps(recMaps, pot, bodies, colliders, RecJoints);
        using (var m = new MemoryStream())
        using (var w = new BinaryWriter(m))
        {
            w.Write(0x31474F47); // "GOG1"
            w.Write(RecordVersion);
            if (recMaps == null) w.Write(-1);
            else NativeBox2D.Geometry(w, recMaps);
            // Index tables: collider paths/components and joint paths, in the indices records use.
            var names = new StringBuilder();
            foreach (var c in colliders) names.Append(EngineDump.PathOf(c.transform)).Append('#').Append(EngineDump.ComponentIndex(c)).Append('\n');
            names.Append('\n');
            foreach (var b in bodies) names.Append(EngineDump.PathOf(b.transform)).Append('\n');
            names.Append('\n');
            foreach (var j in RecJoints) names.Append(EngineDump.PathOf(j.transform)).Append('#').Append(j.GetType().Name).Append('\n');
            var bytes = Encoding.UTF8.GetBytes(names.ToString());
            w.Write(bytes.Length);
            w.Write(bytes);
            return m.ToArray();
        }
    }

    // ---- 'q': every constant and cached struct the controller and friction scripts use, as one JSON row ----
    public string ControllerJson()
    {
        var sb = new StringBuilder();
        EngineDump.Controller(sb);
        return sb.ToString();
    }

    static class R
    {
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        public static readonly FieldInfo OldAngle = typeof(PlayerControl).GetField("oldAngle", Any),
            Motor = typeof(PlayerControl).GetField("motor", Any), Slider = typeof(PlayerControl).GetField("slider", Any);
    }
}

// Harmony hooks feeding the recorder. Each is patched on its own and guarded, so a hook that fails to install (or
// throws) only loses its events, never the game's own behaviour.
static class RecordHooks
{
    public static void Install(Harmony harmony, ManualLogSource log)
    {
        void Hook(Type type, string method, Type[] args, string prefix, string postfix)
        {
            try
            {
                var original = AccessTools.Method(type, method, args) ?? throw new MissingMethodException(type.Name, method);
                harmony.Patch(original, prefix == null ? null : new HarmonyMethod(typeof(RecordHooks), prefix),
                    postfix == null ? null : new HarmonyMethod(typeof(RecordHooks), postfix));
            }
            catch (Exception e)
            {
                log.LogWarning($"recorder hook {type.Name}.{method} not installed: {e.Message}");
            }
        }
        var c = new[] { typeof(Collision2D) };
        Hook(typeof(HammerCollisions), "OnCollisionEnter2D", c, null, nameof(HammerEnter));
        Hook(typeof(HammerCollisions), "OnCollisionStay2D", c, null, nameof(HammerStay));
        Hook(typeof(HammerCollisions), "OnCollisionExit2D", c, null, nameof(HammerExit));
        Hook(typeof(PotSounds), "OnCollisionEnter2D", c, null, nameof(PotEnter));
        Hook(typeof(PotSounds), "OnCollisionStay2D", c, null, nameof(PotStay));
        Hook(typeof(PotSounds), "OnCollisionExit2D", c, null, nameof(PotExit));
        Hook(typeof(PlayerControl), "HammerReturn", Type.EmptyTypes, nameof(ReturnBefore), nameof(ReturnAfter));
        Hook(typeof(Rigidbody2D), nameof(Rigidbody2D.MovePosition), new[] { typeof(Vector2) }, nameof(MovePosition), null);
    }

    static void Guard(Action a)
    {
        if (Game.Recording == null) return;
        try { a(); } catch (Exception) { }
    }

    static void HammerEnter(HammerCollisions __instance, Collision2D coll) => Guard(() => Game.Recording.OnCollision(__instance, Game.EvEnter, coll));
    static void HammerStay(HammerCollisions __instance, Collision2D coll) => Guard(() => Game.Recording.OnCollision(__instance, Game.EvStay, coll));
    static void HammerExit(HammerCollisions __instance, Collision2D coll) => Guard(() => Game.Recording.OnCollision(__instance, Game.EvExit, coll));
    static void PotEnter(PotSounds __instance, Collision2D coll) => Guard(() => Game.Recording.OnCollision(__instance, Game.EvEnter, coll));
    static void PotStay(PotSounds __instance, Collision2D coll) => Guard(() => Game.Recording.OnCollision(__instance, Game.EvStay, coll));
    static void PotExit(PotSounds __instance, Collision2D coll) => Guard(() => Game.Recording.OnCollision(__instance, Game.EvExit, coll));
    static void ReturnBefore() => Guard(() => Game.Recording.OnHammerReturn(false));
    static void ReturnAfter() => Guard(() => Game.Recording.OnHammerReturn(true));
    static void MovePosition(Rigidbody2D __instance, Vector2 position) => Guard(() => Game.Recording.OnMovePosition(__instance, position));
}
