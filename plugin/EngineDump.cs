using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using BepInEx.Logging;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;

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
                .F("awake", rb.IsAwake()).F("material", Mat(rb.sharedMaterial)).F("colliders", rb.attachedColliderCount).End();

        foreach (var j in All<Joint2D>(scene))
        {
            var o = new Json(sb, "joint").F("type", j.GetType().Name).F("path", PathOf(j.transform))
                .F("enabled", j.enabled && j.gameObject.activeInHierarchy)
                .F("connected", j.connectedBody == null ? null : PathOf(j.connectedBody.transform))
                .F("enableCollision", j.enableCollision).F("breakForce", j.breakForce).F("breakTorque", j.breakTorque);
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
            var o = new Json(sb, "collider").F("type", c.GetType().Name).F("path", PathOf(c.transform)).F("layer", c.gameObject.layer)
                .F("active", c.enabled && c.gameObject.activeInHierarchy).F("trigger", c.isTrigger)
                .F("body", c.attachedRigidbody == null ? null : PathOf(c.attachedRigidbody.transform))
                .F("material", Mat(c.sharedMaterial)).F("shapeCount", c.shapeCount).F("offset", c.offset);
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

    // Every loaded scene: RL mode moves the level's physics into scenes of its own.
    static IEnumerable<int> Tracked(Component script, List<Collider2D> cols) =>
        Traverse.Create(script).Field("collisionPoints").GetValue<Dictionary<Collider2D, ContactPoint2D[]>>()?.Keys
            .Select(k => cols.IndexOf(k)) ?? Enumerable.Empty<int>();

    static IEnumerable<T> All<T>(Scene s) where T : Component =>
        Resources.FindObjectsOfTypeAll<T>().Where(x => x.gameObject.scene.IsValid())
            .OrderBy(x => PathOf(x.transform), System.StringComparer.Ordinal);

    internal static string PathOf(Transform t) => t.parent == null ? t.name : PathOf(t.parent) + "/" + t.name;

    static string Mat(PhysicsMaterial2D m) =>
        m == null ? null : $"{m.name} friction={Json.Num(m.friction)} bounciness={Json.Num(m.bounciness)}";

    // Minimal JSON-lines writer; Python's json module accepts the NaN/Infinity tokens.
    sealed class Json
    {
        readonly StringBuilder sb;
        public Json(StringBuilder sb, string kind) { this.sb = sb; sb.Append("{\"kind\":\"").Append(kind).Append('"'); }
        StringBuilder Key(string k) => sb.Append(",\"").Append(k).Append("\":");
        public Json F(string k, string v) { var b = Key(k); if (v == null) b.Append("null"); else b.Append('"').Append(v.Replace("\\", "\\\\").Replace("\"", "\\\"")).Append('"'); return this; }
        public Json F(string k, float v) { Key(k).Append(Num(v)); return this; }
        public Json F(string k, int v) { Key(k).Append(v.ToString(CultureInfo.InvariantCulture)); return this; }
        public Json F(string k, bool v) { Key(k).Append(v ? "true" : "false"); return this; }
        public Json F(string k, Vector2 v) { Key(k).Append('[').Append(Num(v.x)).Append(',').Append(Num(v.y)).Append(']'); return this; }
        public Json F(string k, IEnumerable<float> v) { Key(k).Append('[').Append(string.Join(",", v.Select(Num).ToArray())).Append(']'); return this; }
        public Json F(string k, IEnumerable<int> v) { Key(k).Append('[').Append(string.Join(",", v.Select(i => i.ToString(CultureInfo.InvariantCulture)).ToArray())).Append(']'); return this; }
        public void End() => sb.Append("}\n");
        public static string Num(float v) =>
            float.IsNaN(v) ? "NaN" : float.IsInfinity(v) ? (v > 0 ? "Infinity" : "-Infinity") : v.ToString("R", CultureInfo.InvariantCulture);
    }
}
