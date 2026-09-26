using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

// -goi-colliders: outlines every solid 2D physics shape on top of the main camera's image, so frames show what the
// physics collides with next to what the game draws. Cyan: dynamic bodies (the player, props). Magenta: the rest.
static class ColliderOverlay
{
    static Material material;
    static readonly Dictionary<Collider2D, Vector3[]> staticOutlines = new Dictionary<Collider2D, Vector3[]>();

    public static void Enable()
    {
        material = new Material(Shader.Find("Hidden/Internal-Colored")) { hideFlags = HideFlags.HideAndDontSave };
        material.SetInt("_ZTest", (int)CompareFunction.Always);
        material.SetInt("_ZWrite", 0);
        material.SetInt("_Cull", (int)CullMode.Off);
        Camera.onPostRender += Draw;
    }

    static void Draw(Camera cam)
    {
        if (cam != Camera.main) return;
        var frustum = GeometryUtility.CalculateFrustumPlanes(cam);
        GL.PushMatrix();
        GL.LoadProjectionMatrix(cam.projectionMatrix);
        GL.modelview = cam.worldToCameraMatrix;
        material.SetPass(0);
        GL.Begin(GL.LINES);
        foreach (var c in Object.FindObjectsOfType<Collider2D>())
        {
            if (!c.enabled || c.isTrigger || !GeometryUtility.TestPlanesAABB(frustum, c.bounds)) continue;
            var body = c.attachedRigidbody;
            bool dynamic = body != null && body.bodyType == RigidbodyType2D.Dynamic;
            GL.Color(dynamic ? Color.cyan : Color.magenta);
            // Static shapes never move (a restore re-creates them identically), so their outlines are computed once.
            if (body != null || !staticOutlines.TryGetValue(c, out var lines)) lines = Outline(c);
            if (body == null) staticOutlines[c] = lines;
            foreach (var p in lines) GL.Vertex(p);
        }
        GL.End();
        GL.PopMatrix();
    }

    // The shape's boundary in world space, as line segment pairs: edges that belong to exactly one triangle of the
    // physics engine's own mesh of the shape. Seams between convex pieces of one polygon show too.
    static Vector3[] Outline(Collider2D c)
    {
        var mesh = c.CreateMesh(true, true);
        if (mesh == null) return new Vector3[0];
        var v = mesh.vertices;
        var t = mesh.triangles;
        Object.Destroy(mesh);
        var edges = new Dictionary<Vector4, int>();
        for (int i = 0; i < t.Length; i += 3)
            for (int j = 0; j < 3; j++)
            {
                Vector3 a = v[t[i + j]], b = v[t[i + (j + 1) % 3]];
                var key = a.x < b.x || (a.x == b.x && a.y < b.y) ? new Vector4(a.x, a.y, b.x, b.y) : new Vector4(b.x, b.y, a.x, a.y);
                edges[key] = edges.TryGetValue(key, out int n) ? n + 1 : 1;
            }
        var lines = new List<Vector3>();
        foreach (var e in edges)
            if (e.Value == 1) { lines.Add(new Vector3(e.Key.x, e.Key.y)); lines.Add(new Vector3(e.Key.z, e.Key.w)); }
        return lines.ToArray();
    }
}
