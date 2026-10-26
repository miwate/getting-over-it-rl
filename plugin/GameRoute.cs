using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using FluffyUnderware.Curvy;
using UnityEngine;

// 'u': what a simulator needs to reproduce Observe outside the game (goi_jax/game/route.py, observe.py, reward.py):
// the level route (LevelSpline: per segment its Curvy approximation cache, cached distances and the four points its
// interpolation reads), the colliders behind the flags (lake, Snake, the space zone's trigger), the ray layers, and
// probes: the game's own GetNearestPointTF / TFToDistance / DistanceToTF / Interpolate on deterministic points, to
// check a port against. Derived from game data: never commit it.
partial class Game
{
    public string RouteJson()
    {
        var sb = new StringBuilder();
        var segs = Enumerable.Range(0, route.Count).Select(i => route[i]).ToArray();
        var stepField = typeof(CurvySplineSegment).GetField("mStepSize", BindingFlags.Instance | BindingFlags.NonPublic);
        var o = new Json(sb, "route")
            .F("path", EngineDump.PathOf(route.transform))
            .F("interpolation", route.Interpolation.ToString())
            .F("closed", route.Closed).F("autoEndTangents", route.AutoEndTangents)
            .F("count", route.Count).F("length", route.Length)
            .F("cacheDensity", route.CacheDensity).F("maxPointsPerUnit", route.MaxPointsPerUnit)
            .F("tension", route.Tension).F("continuity", route.Continuity).F("bias", route.Bias)
            .Tf("transform", route.transform)
            .F("ahead", Ahead).F("maxAdvance", MaxAdvance).F("onRoute", OnRoute)
            .F("playerZ", player.transform.position.z);
        var segJson = new List<string>();
        foreach (var s in segs)
        {
            var ssb = new StringBuilder();
            var prev = route.GetPreviousControlPointUsingFollowUp(s);
            var next = route.GetNextControlPointUsingFollowUp(s);
            var next2 = next ? next.Spline.GetNextControlPointUsingFollowUp(next) : null;
            Vector3 p0 = s.TTransform.localPosition;
            Vector3 p1 = next ? next.TTransform.localPosition : p0;
            var j = new Json(ssb, "segment")
                .F("distance", s.Distance).F("length", s.Length).F("cacheSize", s.CacheSize)
                .F("stepSize", stepField == null ? float.NaN : (float)stepField.GetValue(s))
                .F("approximation", s.Approximation.SelectMany(v => new[] { v.x, v.y, v.z }))
                .F("approximationDistances", s.ApproximationDistances)
                .F("p0", p0).F("p1", p1)
                // CatmullRom / TCB: previous and next-but-one control points (the segment's own when missing)
                .F("t0", prev ? prev.TTransform.localPosition : p0)
                .F("t1", next2 ? next2.TTransform.localPosition : p1)
                // Bezier: the handles as interpolateBezier adds them (float Vector3 sums)
                .F("bezierT0", p0 + s.HandleOut).F("bezierT1", next ? p1 + next.HandleIn : p1)
                .F("hasNext", (bool)next);
            j.End(); // appends "}\n": the newline is trimmed for nesting
            segJson.Add(ssb.ToString().TrimEnd('\n'));
        }
        o.Raw("segments", "[" + string.Join(",", segJson) + "]");

        // Flags: the colliders they come from, as "path#component" (the geometry record's collider names).
        string Name(Collider2D c) => EngineDump.PathOf(c.transform) + "#" + EngineDump.ComponentIndex(c);
        o.F("lakeColliders", lake.GetComponents<Collider2D>().Select(Name))
         .F("lakeBodyColliders", lake.GetComponent<Rigidbody2D>() == null ? new string[0]
             : lake.GetComponent<Rigidbody2D>().GetComponentsInChildren<Collider2D>(true).Where(c => c.attachedRigidbody == lake.GetComponent<Rigidbody2D>()).Select(Name))
         .F("snakeCollider", Name(snake));
        var zone = gravity.GetComponent<BoxCollider2D>();
        o.F("zeroGravCollider", Name(zone)).F("zeroGravMin", (Vector2)zone.bounds.min).F("zeroGravMax", (Vector2)zone.bounds.max);
        o.F("layerDefault", LayerMask.NameToLayer("Default")).F("layerTerrain", LayerMask.NameToLayer("Terrain"))
         .F("layerPlayer", LayerMask.NameToLayer("Player")).F("solidMask", (int)solid.layerMask)
         .F("queriesStartInColliders", Physics2D.queriesStartInColliders);
        o.F("potCollidersTouch", pot.GetComponentsInChildren<Collider2D>(true).Where(c => c.attachedRigidbody == pot).Select(Name))
         .F("tipColliders", tip.GetComponentsInChildren<Collider2D>(true).Where(c => c.attachedRigidbody == tip).Select(Name));

        // Probes. Nearest: points scattered around the route (a random cache point plus up to 8 either way, z = the
        // player's), each with the game's GetNearestPointTF and TFToDistance of it. Distance: arc lengths over the
        // whole route and a little beyond, each with DistanceToTF and Interpolate of that.
        var rng = new System.Random(12345);
        var near = new List<float>();
        float z = player.transform.position.z;
        for (int i = 0; i < 6000; i++)
        {
            var s = segs[rng.Next(segs.Length)];
            var a = s.Approximation[rng.Next(s.Approximation.Length)];
            var p = new Vector3(a.x + (float)(rng.NextDouble() * 16 - 8), a.y + (float)(rng.NextDouble() * 16 - 8), z);
            float tf = route.GetNearestPointTF(p);
            near.AddRange(new[] { p.x, p.y, p.z, tf, route.TFToDistance(tf) });
        }
        o.F("probeNearest", near);
        var dist = new List<float>();
        float length = route.Length;
        for (int i = 0; i < 3000; i++)
        {
            float d = i < 2900 ? (float)(rng.NextDouble() * (length + 10) - 5) : i < 2950 ? segs[i % segs.Length].Distance : length - (i - 2950) * 0.01f;
            float tf = route.DistanceToTF(d);
            Vector3 q = route.Interpolate(tf);
            dist.AddRange(new[] { d, tf, q.x, q.y, q.z });
        }
        o.F("probeDistance", dist);
        o.End();
        return sb.ToString();
    }
}
