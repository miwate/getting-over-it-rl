using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using BepInEx;
using UnityEngine;
using UnityEngine.SceneManagement;

// -goi-map <name> (RL mode): loads maps/<name>.txt (the build deploys maps/ to BepInEx/plugins/goi-maps; a file path
// works too). A map edits the level as it loads, before Game lists the level's physics objects, so every restore
// rebuilds the edits like the rest of the level: new objects are the last children of Mountain, in file order.
// One command per line, '#' starts a comment, level coordinates in metres:
//   look <path>             later rocks use this level object's material (resolve it before hiding the object)
//   off <path>              deactivate a level object (e.g. Water, the lake's surface)
//   hide x0 y0 x1 y1        deactivate the static terrain whose collider centre is in the box (its visual too)
//   rock x1 y1 x2 y2 ...    a convex Superfriction rock, extruded 3 m deep around the play plane
//   clone <path> x y        a copy of a level object (e.g. Mountain/Anvil) whose collider's bottom centre is at x y
class MapLoader
{
    public static string Dir => Path.Combine(Paths.PluginPath, "goi-maps");

    public static void Load(Scene scene, string map)
    {
        string path = File.Exists(map) ? map : Path.Combine(Dir, map + ".txt");
        var loader = new MapLoader(scene);
        var lines = File.ReadAllLines(path);
        for (int i = 0; i < lines.Length; i++)
        {
            var f = lines[i].Split('#')[0].Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
            if (f.Length == 0) continue;
            try { loader.Run(f); }
            catch (Exception e) { throw new FormatException($"{path}:{i + 1}: {lines[i].Trim()}: {e.Message}", e); }
        }
    }

    readonly Transform mountain;
    readonly float z; // the play plane's depth (the climber's)
    readonly Collider2D[] terrain;
    readonly PhysicsMaterial2D superfriction;
    Material look;

    MapLoader(Scene scene)
    {
        var roots = scene.GetRootGameObjects();
        mountain = roots.Single(g => g.name == "Mountain").transform;
        z = roots.Single(g => g.name == "Player").transform.position.z;
        terrain = roots.SelectMany(g => g.GetComponentsInChildren<Collider2D>())
            .Where(c => c.gameObject.layer == 10 && c.attachedRigidbody == null && !c.isTrigger).ToArray();
        superfriction = terrain.Select(c => c.sharedMaterial).First(m => m != null && m.name == "Superfriction");
    }

    void Run(string[] f)
    {
        float N(int i) => float.Parse(f[i], CultureInfo.InvariantCulture);
        switch (f[0])
        {
            case "look": look = Find(f[1]).GetComponentInChildren<Renderer>().sharedMaterial; break;
            case "off": Find(f[1]).SetActive(false); break;
            case "hide": Hide(Rect.MinMaxRect(N(1), N(2), N(3), N(4))); break;
            case "rock": Rock(Enumerable.Range(0, (f.Length - 1) / 2).Select(i => new Vector2(N(1 + 2 * i), N(2 + 2 * i))).ToArray()); break;
            case "clone": Clone(Find(f[1]), N(2), N(3)); break;
            default: throw new FormatException($"unknown command '{f[0]}'");
        }
    }

    static GameObject Find(string path) => GameObject.Find(path) ?? throw new ArgumentException($"no active object '{path}'");

    void Hide(Rect box)
    {
        foreach (var c in terrain) if (box.Contains(c.bounds.center)) c.gameObject.SetActive(false);
    }

    void Clone(GameObject src, float x, float y)
    {
        var b = src.GetComponent<Collider2D>().bounds;
        var copy = UnityEngine.Object.Instantiate(src, src.transform.parent);
        copy.transform.position = src.transform.position + new Vector3(x - b.center.x, y - b.min.y, 0f);
    }

    // ponytail: convex polygons only (fan caps); a concave rock = several convex ones.
    void Rock(Vector2[] p)
    {
        if (look == null) throw new InvalidOperationException("'look' must come before the first rock");
        float area = 0;
        for (int i = 0; i < p.Length; i++) area += p[i].x * p[(i + 1) % p.Length].y - p[(i + 1) % p.Length].x * p[i].y;
        if (area < 0) Array.Reverse(p); // counter-clockwise, so the faces below point outward
        var g = new GameObject("MapRock") { layer = 10 };
        g.transform.SetParent(mountain, true); // world transform stays identity: points are level coordinates
        var col = g.AddComponent<PolygonCollider2D>();
        col.sharedMaterial = superfriction;
        col.points = p;
        g.AddComponent<MeshFilter>().sharedMesh = Extrude(p, z - 1.5f, z + 1.5f);
        g.AddComponent<MeshRenderer>().sharedMaterial = look;
    }

    // The front cap (the camera looks along +z) and the sides of a counter-clockwise convex polygon.
    static Mesh Extrude(Vector2[] p, float front, float back)
    {
        var v = new List<Vector3>();
        var t = new List<int>();
        for (int i = 0; i < p.Length; i++) v.Add(new Vector3(p[i].x, p[i].y, front));
        for (int i = 1; i < p.Length - 1; i++) t.AddRange(new[] { 0, i + 1, i });
        for (int i = 0; i < p.Length; i++)
        {
            Vector2 a = p[i], c = p[(i + 1) % p.Length];
            int k = v.Count;
            v.AddRange(new[] { new Vector3(a.x, a.y, front), new Vector3(c.x, c.y, front), new Vector3(c.x, c.y, back), new Vector3(a.x, a.y, back) });
            t.AddRange(new[] { k, k + 1, k + 2, k, k + 2, k + 3 });
        }
        var mesh = new Mesh { vertices = v.ToArray(), triangles = t.ToArray(), uv = v.Select(q => new Vector2(q.x + q.z, q.y) * 0.25f).ToArray() };
        mesh.RecalculateNormals();
        return mesh;
    }
}
