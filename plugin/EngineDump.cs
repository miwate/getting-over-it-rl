using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using BepInEx.Logging;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

// One-time dump of what only the native physics engine knows: live settings, engine-computed mass properties,
// and the convex pieces Unity actually built from each collider. Output is derived from game data: never commit it.
static class EngineDump
{
    public static void Write(string file, ManualLogSource log)
    {
        var scene = SceneManager.GetActiveScene();
        var sb = new StringBuilder();

        new Json(sb, "settings").F("unity", Application.unityVersion).F("scene", scene.name)
            .F("fixedDeltaTime", Time.fixedDeltaTime).F("maximumDeltaTime", Time.maximumDeltaTime).F("timeScale", Time.timeScale)
            .F("gravity", Physics2D.gravity).F("velocityIterations", Physics2D.velocityIterations).F("positionIterations", Physics2D.positionIterations)
            .F("velocityThreshold", Physics2D.velocityThreshold).F("maxLinearCorrection", Physics2D.maxLinearCorrection)
            .F("maxAngularCorrection", Physics2D.maxAngularCorrection).F("maxTranslationSpeed", Physics2D.maxTranslationSpeed)
            .F("maxRotationSpeed", Physics2D.maxRotationSpeed).F("baumgarteScale", Physics2D.baumgarteScale)
            .F("baumgarteTOIScale", Physics2D.baumgarteTOIScale).F("timeToSleep", Physics2D.timeToSleep)
            .F("linearSleepTolerance", Physics2D.linearSleepTolerance).F("angularSleepTolerance", Physics2D.angularSleepTolerance)
            .F("defaultContactOffset", Physics2D.defaultContactOffset).F("simulationMode", Physics2D.simulationMode.ToString())
            .F("queriesHitTriggers", Physics2D.queriesHitTriggers).F("queriesStartInColliders", Physics2D.queriesStartInColliders)
            .F("callbacksOnDisable", Physics2D.callbacksOnDisable).F("reuseCollisionCallbacks", Physics2D.reuseCollisionCallbacks)
            .F("autoSyncTransforms", Physics2D.autoSyncTransforms).F("physics3dAutoSimulation", Physics.autoSimulation)
            .F("layerCollisionMasks", Enumerable.Range(0, 32).Select(Physics2D.GetLayerCollisionMask))
            .F("runInBackground", Application.runInBackground).F("targetFrameRate", Application.targetFrameRate)
            .F("vSyncCount", QualitySettings.vSyncCount).F("qualityLevel", QualitySettings.GetQualityLevel())
            .F("screen", $"{Screen.width}x{Screen.height} fullscreen={Screen.fullScreen}").End();

        // The player controller's scene values a simulation of it needs (PlayerControl, HammerCollisions, PotSounds): the
        // mouse acceleration curve sampled every 0.05 of input up to 25, the dead zone and epsilons, the cursor body,
        // both friction materials (stuck / sliding) of the hammer's head and the pot's bottom, where the two friction
        // scripts sit (which collisions they hear) and the layer whose collisions they count.
        var cols = All<Collider2D>(scene).ToList(); // the collider rows below, in order
        var control = Object.FindObjectOfType<PlayerControl>();
        if (control != null)
        {
            var t = Traverse.Create(control);
            var hc = Object.FindObjectOfType<HammerCollisions>();
            var ps = Object.FindObjectOfType<PotSounds>();
            new Json(sb, "controller").F("deadzone", control.deadzone).F("trackpad", control.trackpad)
                .F("mouseSensitivity", t.Field("mouseSensitivity").GetValue<float>())
                .F("posEpsilon", t.Field("posEpsilon").GetValue<float>()).F("angleEpsilon", t.Field("angleEpsilon").GetValue<float>())
                .F("mouseCurveStep", 0.05f).F("mouseCurve", Enumerable.Range(0, 501).Select(i => control.mouseCurve.Evaluate(i * 0.05f)))
                .F("mouseCurveKeys", control.mouseCurve.keys.SelectMany(k => new[] { k.time, k.value, k.inTangent, k.outTangent }))
                .F("cursor", PathOf(control.fakeCursor)).F("hinge", PathOf(control.hj.transform)).F("slider", PathOf(control.sj.transform))
                .F("hammerStatic", Mat(hc?.staticFriction)).F("hammerSliding", Mat(hc?.slidingFriction))
                .F("potStatic", Mat(ps?.staticFriction)).F("potSliding", Mat(ps?.slidingFriction))
                .F("potBottom", ps?.bottomCol == null ? null : PathOf(ps.bottomCol.transform))
                .F("potSounds", ps == null ? null : PathOf(ps.transform)).F("hammerCollisions", hc == null ? null : PathOf(hc.transform))
                .F("terrainLayer", LayerMask.NameToLayer("Terrain"))
                // their state now (snapshots carry it): sliding or stuck, and the terrain colliders each tracks, as
                // indices of this dump's collider rows; and whether the controller skips its next input
                .F("hammerSlide", Traverse.Create(hc).Field("slide").GetValue<bool>())
                .F("potSlide", Traverse.Create(ps).Field("slide").GetValue<bool>())
                .F("hammerTracked", Tracked(hc, cols)).F("potTracked", Tracked(ps, cols))
                .F("mouseSnap", t.Field("mouseSnap").GetValue<bool>()).F("potZ", control.transform.position.z)
                .F("mouseInput", t.Field("mouseInput").GetValue<Vector2>())
                .F("mouseVelocityAverage", t.Field("mouseVelocityAverage").GetValue<float>())
                .F("mw", t.Field("mw").GetValue<Vector2>()).End();
        }

        // The route the progress is measured along (ProgressMeter's CurvySpline, at the origin, unrotated), as its
        // GetNearestPointTF / TFToDistance / DistanceToTF read it: each segment's start distance, its approximation
        // polyline (x, y, z per point; local F = index * step) and the polyline's distances from the segment start.
        var meter = Object.FindObjectOfType<ProgressMeter>();
        var route = meter == null ? null : meter.GetComponent<FluffyUnderware.Curvy.CurvySpline>();
        if (route != null)
            for (int i = 0; i < route.Count; i++)
            {
                var seg = route[i];
                new Json(sb, "route").F("segment", i).F("distance", seg.Distance).F("length", seg.Length)
                    .F("step", Traverse.Create(seg).Field("mStepSize").GetValue<float>())
                    .F("approx", seg.Approximation.SelectMany(p => new[] { p.x, p.y, p.z }))
                    .F("approxDistances", seg.ApproximationDistances).End();
            }

        foreach (var rb in All<Rigidbody2D>(scene))
            new Json(sb, "body").F("path", PathOf(rb.transform)).F("active", rb.gameObject.activeInHierarchy).F("layer", rb.gameObject.layer)
                .F("bodyType", rb.bodyType.ToString()).F("simulated", rb.simulated).F("useAutoMass", rb.useAutoMass)
                .F("mass", rb.mass).F("inertia", rb.inertia).F("centerOfMass", rb.centerOfMass).F("worldCenterOfMass", rb.worldCenterOfMass)
                .F("drag", rb.drag).F("angularDrag", rb.angularDrag).F("gravityScale", rb.gravityScale)
                .F("interpolation", rb.interpolation.ToString()).F("collisionDetection", rb.collisionDetectionMode.ToString())
                .F("sleepMode", rb.sleepMode.ToString()).F("constraints", rb.constraints.ToString())
                .F("position", rb.position).F("rotation", rb.rotation).F("velocity", rb.velocity).F("angularVelocity", rb.angularVelocity)
                .F("awake", rb.IsAwake()).F("material", Mat(rb.sharedMaterial)).F("colliders", rb.attachedColliderCount)
                .Tf("tf", rb.transform).End();

        foreach (var j in All<Joint2D>(scene))
        {
            var o = new Json(sb, "joint").F("type", j.GetType().Name).F("path", PathOf(j.transform))
                .F("enabled", j.enabled && j.gameObject.activeInHierarchy)
                .F("connected", j.connectedBody == null ? null : PathOf(j.connectedBody.transform))
                .F("enableCollision", j.enableCollision).F("breakForce", j.breakForce).F("breakTorque", j.breakTorque)
                .Tf("tf", j.transform).Tf("connectedTf", j.connectedBody == null ? null : j.connectedBody.transform);
            if (j is AnchoredJoint2D a)
                o.F("anchor", a.anchor).F("connectedAnchor", a.connectedAnchor).F("autoConfigureConnectedAnchor", a.autoConfigureConnectedAnchor);
            if (j is HingeJoint2D h)
                o.F("useMotor", h.useMotor).F("motorSpeed", h.motor.motorSpeed).F("maxMotorTorque", h.motor.maxMotorTorque)
                    .F("useLimits", h.useLimits).F("limits", new[] { h.limits.min, h.limits.max }).F("referenceAngle", h.referenceAngle)
                    .F("jointAngle", h.jointAngle).F("jointSpeed", h.jointSpeed);
            if (j is SliderJoint2D s)
                o.F("angle", s.angle).F("autoConfigureAngle", s.autoConfigureAngle).F("useMotor", s.useMotor)
                    .F("motorSpeed", s.motor.motorSpeed).F("maxMotorForce", s.motor.maxMotorTorque).F("useLimits", s.useLimits)
                    .F("limits", new[] { s.limits.min, s.limits.max }).F("referenceAngle", s.referenceAngle)
                    .F("jointTranslation", s.jointTranslation).F("jointSpeed", s.jointSpeed);
            if (j is FixedJoint2D f)
                o.F("frequency", f.frequency).F("dampingRatio", f.dampingRatio).F("referenceAngle", f.referenceAngle);
            o.End();
        }

        foreach (var c in cols)
        {
            var o = new Json(sb, "collider").F("type", c.GetType().Name).F("path", PathOf(c.transform)).F("component", ComponentIndex(c))
                .F("layer", c.gameObject.layer)
                .F("active", c.enabled && c.gameObject.activeInHierarchy).F("trigger", c.isTrigger)
                .F("body", c.attachedRigidbody == null ? null : PathOf(c.attachedRigidbody.transform))
                .F("material", Mat(c.sharedMaterial)).F("shapeCount", c.shapeCount).F("offset", c.offset).Tf("tf", c.transform);
            if (c is CircleCollider2D ci) o.F("radius", ci.radius);
            if (c is BoxCollider2D bx) o.F("size", bx.size).F("edgeRadius", bx.edgeRadius);
            if (c is PolygonCollider2D pc) o.F("pathCount", pc.pathCount).F("points", pc.GetTotalPointCount());
            var mesh = c.shapeCount > 0 ? c.CreateMesh(true, true) : null; // world-space triangles of the engine's shapes
            if (mesh != null)
            {
                o.F("verts", mesh.vertices.SelectMany(v => new[] { v.x, v.y })).F("tris", mesh.triangles);
                Object.Destroy(mesh);
            }
            o.End();
        }

        File.WriteAllText(file, sb.ToString());
        log.LogInfo($"engine dump written: {file}");
    }

    // The 'q' command's row: the serialized fields and cached structs of PlayerControl, HammerCollisions and PotSounds
    // (read by reflection), Unity's own AnimationCurve.Evaluate sampled on grids (to test a port of it), and a probe
    // of how this Mono JIT evaluates float arithmetic (float32 per operation, or wider intermediates).
    internal static void Controller(StringBuilder sb)
    {
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        object Get(object o, string name) => o?.GetType().GetField(name, Any)?.GetValue(o);
        var pc = All<PlayerControl>(default).FirstOrDefault();
        var hc = All<HammerCollisions>(default).FirstOrDefault();
        var ps = All<PotSounds>(default).FirstOrDefault();
        var o = new Json(sb, "controller");
        if (pc == null) { o.F("error", "no PlayerControl").End(); return; }
        string Motor(object m) => m is JointMotor2D j ? $"{{\"motorSpeed\":{Json.Num(j.motorSpeed)},\"maxMotorTorque\":{Json.Num(j.maxMotorTorque)}}}" : "null";
        string MatJ(object m) => m is PhysicsMaterial2D p
            ? $"{{\"name\":\"{p.name}\",\"friction\":{Json.Num(p.friction)},\"bounciness\":{Json.Num(p.bounciness)}}}" : "null";
        o.F("path", PathOf(pc.transform))
            .F("deadzone", pc.deadzone).F("mouseSensitivity", (float)Get(pc, "mouseSensitivity"))
            .F("posEpsilon", (float)Get(pc, "posEpsilon")).F("angleEpsilon", (float)Get(pc, "angleEpsilon"))
            .F("trackpad", pc.trackpad).F("pauseInputTimer", (float)Get(pc, "pauseInputTimer"))
            .F("inputsToSkip", (int)Get(pc, "inputsToSkip")).F("loadFinished", pc.loadFinished)
            .F("mouseSnap", (bool)Get(pc, "mouseSnap")).F("mouseInput", (Vector2)Get(pc, "mouseInput"))
            .F("oldMouse", (Vector2)Get(pc, "oldMouse")).F("mouseVelocityAverage", (float)Get(pc, "mouseVelocityAverage"))
            .F("mw", (Vector2)Get(pc, "mw")).F("oldAngle", (float)Get(pc, "oldAngle"))
            .Raw("motor", Motor(Get(pc, "motor"))).Raw("slider", Motor(Get(pc, "slider")))
            .Raw("hjMotor", Motor(pc.hj.motor)).Raw("sjMotor", Motor(pc.sj.motor))
            .F("hjPath", PathOf(pc.hj.transform)).F("hjConnected", pc.hj.connectedBody == null ? null : PathOf(pc.hj.connectedBody.transform))
            .F("hjUseMotor", pc.hj.useMotor).F("hjUseLimits", pc.hj.useLimits).F("hjLimits", new[] { pc.hj.limits.min, pc.hj.limits.max })
            .F("hjReferenceAngle", pc.hj.referenceAngle).F("hjJointAngle", pc.hj.jointAngle)
            .F("sjPath", PathOf(pc.sj.transform)).F("sjConnected", pc.sj.connectedBody == null ? null : PathOf(pc.sj.connectedBody.transform))
            .F("sjUseMotor", pc.sj.useMotor).F("sjUseLimits", pc.sj.useLimits).F("sjLimits", new[] { pc.sj.limits.min, pc.sj.limits.max })
            .F("sjAngle", pc.sj.angle).F("sjReferenceAngle", pc.sj.referenceAngle).F("sjJointTranslation", pc.sj.jointTranslation)
            .F("tipPath", PathOf(pc.tip)).F("cursorPath", PathOf(pc.fakeCursor));
        var cursor = pc.fakeCursor.GetComponent<Rigidbody2D>();
        if (cursor != null)
            o.F("cursorBodyType", cursor.bodyType.ToString()).F("cursorInterpolation", cursor.interpolation.ToString())
                .F("cursorConstraints", cursor.constraints.ToString()).F("cursorColliders", cursor.attachedColliderCount);
        foreach (var name in new[] { "mouseCurve", "trackpadCurve", "stickCurve" })
            o.Raw(name, Curve(Get(pc, name) as AnimationCurve));
        if (hc != null)
        {
            var col = hc.GetComponent<Collider2D>();
            o.Raw("hammerStatic", MatJ(hc.staticFriction)).Raw("hammerSliding", MatJ(hc.slidingFriction))
                .F("hammerMoveThreshold", (float)Get(hc, "moveThreshold")).F("hammerHardHitThreshold", (float)Get(hc, "hardHitThreshold"))
                .F("hammerWhooshThreshold", (float)Get(hc, "whooshThreshold"))
                .F("hammerContactBuffer", (Get(hc, "contacts") as ContactPoint2D[])?.Length ?? -1)
                .F("hammerSlide", (bool)Get(hc, "slide")).F("hammerPath", PathOf(hc.transform))
                .F("hammerCollider", col == null ? null : PathOf(col.transform)).F("hammerColliderType", col?.GetType().Name)
                .Raw("hammerCurrent", MatJ(col?.sharedMaterial)).F("hammerShapeCount", col == null ? -1 : col.shapeCount);
        }
        if (ps != null)
            o.Raw("potStatic", MatJ(ps.staticFriction)).Raw("potSliding", MatJ(ps.slidingFriction))
                .F("potMoveThreshold", (float)Get(ps, "moveThreshold")).F("potRollThreshold", (float)Get(ps, "rollThreshold"))
                .F("potHardClunkThreshold", (float)Get(ps, "hardClunkThreshold"))
                .F("potContactBuffer", (Get(ps, "contacts") as ContactPoint2D[])?.Length ?? -1)
                .F("potSlide", (bool)Get(ps, "slide")).F("potSoundsPath", PathOf(ps.transform))
                .F("potBottomCollider", ps.bottomCol == null ? null : PathOf(ps.bottomCol.transform))
                .F("potBottomComponent", ps.bottomCol == null ? -1 : ComponentIndex(ps.bottomCol))
                .Raw("potBottomCurrent", MatJ(ps.bottomCol?.sharedMaterial))
                .F("potRB", ps.potRB == null ? null : PathOf(ps.potRB.transform));
        o.F("fixedDeltaTime", Time.fixedDeltaTime).F("gravity", Physics2D.gravity)
            .Raw("monoFloatProbe", FloatProbe())
            .F("scriptFields", Game.ScriptFields).End();
    }

    // Unity's keys, wrap modes and its own Evaluate on grids: |mouse input| (sensitivity 1) runs 0..100 here.
    static string Curve(AnimationCurve c)
    {
        if (c == null) return "null";
        var e = new StringBuilder("{\"preWrapMode\":\"").Append(c.preWrapMode).Append("\",\"postWrapMode\":\"").Append(c.postWrapMode)
            .Append("\",\"keys\":[");
        var keys = c.keys;
        for (int i = 0; i < keys.Length; i++)
        {
            var k = keys[i];
            e.Append(i == 0 ? "" : ",").Append("{\"time\":").Append(Json.Num(k.time)).Append(",\"value\":").Append(Json.Num(k.value))
                .Append(",\"inTangent\":").Append(Json.Num(k.inTangent)).Append(",\"outTangent\":").Append(Json.Num(k.outTangent))
                .Append(",\"inWeight\":").Append(Json.Num(k.inWeight)).Append(",\"outWeight\":").Append(Json.Num(k.outWeight))
                .Append(",\"weightedMode\":\"").Append(k.weightedMode).Append("\"}");
        }
        var xs = new List<float>();
        for (int i = 0; i <= 4000; i++) xs.Add(i * 0.0025f);      // 0..10, fine
        for (int i = 1; i <= 360; i++) xs.Add(10f + i * 0.25f);   // 10..100
        foreach (var x in new[] { -1f, -0.001f, 1e-6f, 1e-3f, 150f, 1000f }) xs.Add(x);
        if (keys.Length > 0)
            foreach (var k in keys) { xs.Add(k.time); xs.Add(BitConverter.ToSingle(BitConverter.GetBytes(BitConverter.ToInt32(BitConverter.GetBytes(k.time), 0) + 1), 0)); }
        e.Append("],\"sampleX\":[").Append(string.Join(",", xs.Select(Json.Num).ToArray())).Append("],\"sampleY\":[")
            .Append(string.Join(",", xs.Select(x => Json.Num(c.Evaluate(x))).ToArray())).Append("]}");
        return e.ToString();
    }

    // Values that come out differently under float32-per-operation and wider (double) intermediates. Inputs come from
    // an array so the compiler can't fold them; the results are stored to float locals, as in game code.
    static readonly float[] probeIn = { 16777216f, 1f, 0.1f, 3f, 1e-8f, 1.0000001f };
    static string FloatProbe()
    {
        var p = probeIn;
        float a = (p[0] + p[1]) - p[0];          // f32: 0, double: 1
        float b = p[2] * p[3] - 0.3f;            // f32 product rounds first
        float c = (p[1] + p[4]) - p[1];          // f32: 0, double: 1e-8
        float d = p[5] * p[5] * p[5] * p[5];     // chained products
        float e = Mathf.Sqrt(p[2] * p[2] + p[3] * p[3]);
        var v = new Vector2(p[0], 0f) + new Vector2(p[1], 0f);
        float g = v.x - p[0];                    // a struct field store rounds in either mode
        return $"{{\"sum\":{Json.Num(a)},\"mulSub\":{Json.Num(b)},\"tiny\":{Json.Num(c)},\"pow4\":{Json.Num(d)},\"hypot\":{Json.Num(e)},\"structSum\":{Json.Num(g)}}}";
    }

    // Every loaded scene: RL mode moves the level's physics into scenes of its own.
    static IEnumerable<int> Tracked(Component script, List<Collider2D> cols) =>
        Traverse.Create(script).Field("collisionPoints").GetValue<Dictionary<Collider2D, ContactPoint2D[]>>()?.Keys
            .Select(k => cols.IndexOf(k)) ?? Enumerable.Empty<int>();

    internal static IEnumerable<T> All<T>(Scene s) where T : Component =>
        Resources.FindObjectsOfTypeAll<T>().Where(x => x.gameObject.scene.IsValid())
            .OrderBy(x => PathOf(x.transform), System.StringComparer.Ordinal);

    // Which of its GameObject's colliders this is: paths alone are ambiguous where an object has several.
    internal static int ComponentIndex(Collider2D c) => System.Array.IndexOf(c.GetComponents<Collider2D>(), c);

    internal static string PathOf(Transform t) => t.parent == null ? t.name : PathOf(t.parent) + "/" + t.name;

    internal static string Mat(PhysicsMaterial2D m) =>
        m == null ? null : $"{m.name} friction={Json.Num(m.friction)} bounciness={Json.Num(m.bounciness)}";
}

// Minimal JSON-lines writer; Python's json module accepts the NaN/Infinity tokens.
sealed class Json
{
    readonly StringBuilder sb;
    public Json(StringBuilder sb, string kind) { this.sb = sb; sb.Append("{\"kind\":\"").Append(kind).Append('"'); }
    StringBuilder Key(string k) => sb.Append(",\"").Append(k).Append("\":");
    static string Str(string v) => v == null ? "null" : "\"" + v.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    public Json F(string k, string v) { Key(k).Append(Str(v)); return this; }
    public Json F(string k, float v) { Key(k).Append(Num(v)); return this; }
    public Json F(string k, int v) { Key(k).Append(v.ToString(CultureInfo.InvariantCulture)); return this; }
    public Json F(string k, bool v) { Key(k).Append(v ? "true" : "false"); return this; }
    public Json F(string k, Vector2 v) { Key(k).Append('[').Append(Num(v.x)).Append(',').Append(Num(v.y)).Append(']'); return this; }
    public Json F(string k, IEnumerable<float> v) { Key(k).Append('[').Append(string.Join(",", v.Select(Num).ToArray())).Append(']'); return this; }
    public Json F(string k, Vector3 v) { Key(k).Append('[').Append(Num(v.x)).Append(',').Append(Num(v.y)).Append(',').Append(Num(v.z)).Append(']'); return this; }
    public Json F(string k, Quaternion q) { Key(k).Append('[').Append(Num(q.x)).Append(',').Append(Num(q.y)).Append(',').Append(Num(q.z)).Append(',').Append(Num(q.w)).Append(']'); return this; }
    public Json F(string k, long v) { Key(k).Append(v.ToString(CultureInfo.InvariantCulture)); return this; }
    public Json F(string k, IEnumerable<string> v) { Key(k).Append('[').Append(string.Join(",", v.Select(Str).ToArray())).Append(']'); return this; }
    public Json Raw(string k, string json) { Key(k).Append(json); return this; } // a nested value already in JSON
    // Pose and scale of a Transform: Unity bakes colliders, mass and joint anchors from the full 3D Transform.
    public Json Tf(string prefix, Transform t) =>
        t == null ? this : F(prefix + "Position", t.position).F(prefix + "Rotation", t.rotation)
            .F(prefix + "LocalScale", t.localScale).F(prefix + "LossyScale", t.lossyScale);
    public Json F(string k, IEnumerable<int> v) { Key(k).Append('[').Append(string.Join(",", v.Select(i => i.ToString(CultureInfo.InvariantCulture)).ToArray())).Append(']'); return this; }
    public void End() => sb.Append("}\n");
    public static string Num(float v) =>
        float.IsNaN(v) ? "NaN" : float.IsInfinity(v) ? (v > 0 ? "Infinity" : "-Infinity") : v.ToString("R", CultureInfo.InvariantCulture);
}
