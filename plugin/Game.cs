using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

// Lockstep control of the loaded level. The game's own controller and friction scripts still do the work: they are
// disabled so Unity stops calling their FixedUpdate, and Step() calls it instead, in the game's script execution
// order, then simulates one physics step of the level's own physics world (see Restore). Unity still sends collision
// callbacks to disabled scripts, so their contact bookkeeping keeps working. Replays from a snapshot are bit-exact.
class Game
{
    public const int ObsFields = 15; // layout: OBS_FIELDS in goi_env/client.py, then RayCount ray distances
    public int K = 4;                // physics steps per action
    public int RayCount = 16;
    public float RayLength = 10f;
    public bool Render { get; private set; } = true;

    readonly float dt = Time.fixedDeltaTime;
    readonly GameObject player;
    readonly PlayerControl pc;
    readonly HammerCollisions hammer;
    readonly PotSounds potSounds;
    readonly GravityControl gravity;
    readonly RestartOnContact lake;
    readonly GiftSpawner giftSpawner;
    readonly OpenGift gift;
    readonly CameraControl cam;
    readonly CraneMove craneMove;
    readonly Rigidbody2D pot, hub, arm, tip, cursor, crane;
    readonly Vector2 craneStart;
    readonly Rigidbody2D[] bodies;       // every body in the level, in a fixed order
    readonly GameObject[] physicsRoots;  // every root holding 2D physics; moved into a fresh physics world on restore
    readonly GameObject[] physicsObjects; // top-most objects with 2D physics, in path order: re-created in this order
    readonly Action hammerFixedUpdate, potFixedUpdate, controlFixedUpdate, poseLateUpdate;
    readonly PoseControl pose;
    readonly ContactFilter2D solid;
    readonly Snapshot start;
    readonly Dictionary<int, Snapshot> saves = new Dictionary<int, Snapshot>(); // ponytail: never freed (~1 KB each); add a drop command if a run saves millions
    Camera[] cameras;
    Scene worldScene;
    PhysicsScene2D world;
    int generation;
    float simTime;
    bool ready;

    // Runs when the level has loaded but before any Start(), so the bodies are still in the authored start pose.
    public Game()
    {
        player = GameObject.Find("Player");
        pc = player.GetComponent<PlayerControl>();
        pot = player.GetComponent<Rigidbody2D>();
        hub = pc.hj.attachedRigidbody;
        arm = pc.hj.connectedBody;
        tip = pc.tip.GetComponent<Rigidbody2D>();
        cursor = pc.fakeCursor.GetComponent<Rigidbody2D>();
        hammer = pc.tip.GetComponent<HammerCollisions>();
        potSounds = player.GetComponentInChildren<PotSounds>();
        gravity = Object.FindObjectOfType<GravityControl>();
        lake = Object.FindObjectOfType<RestartOnContact>();
        giftSpawner = Object.FindObjectOfType<GiftSpawner>();
        cam = Object.FindObjectOfType<CameraControl>();
        craneMove = Object.FindObjectOfType<CraneMove>();
        crane = craneMove.GetComponent<Rigidbody2D>();
        craneStart = crane.position;
        bodies = InLevel<Rigidbody2D>();
        gift = InLevel<OpenGift>().Single(); // inactive until the gift spawns
        physicsRoots = player.scene.GetRootGameObjects()
            .Where(r => r.GetComponentInChildren<Collider2D>(true) != null || r.GetComponentInChildren<Rigidbody2D>(true) != null).ToArray();
        var withPhysics = InLevel<Collider2D>().Select(c => c.gameObject).Concat(bodies.Select(b => b.gameObject)).Distinct().ToArray();
        physicsObjects = withPhysics.Where(g => !withPhysics.Any(a => a != g && g.transform.IsChildOf(a.transform)))
            .OrderBy(g => HierarchyKey(g.transform), StringComparer.Ordinal).ToArray();
        pose = player.GetComponentInChildren<PoseControl>();
        hammerFixedUpdate = Method(hammer, "FixedUpdate");
        potFixedUpdate = Method(potSounds, "FixedUpdate");
        controlFixedUpdate = Method(pc, "FixedUpdate");
        poseLateUpdate = Method(pose, "LateUpdate");
        solid = new ContactFilter2D { useTriggers = false, useLayerMask = true, layerMask = LayerMask.GetMask("Default", "Terrain") };
        start = Capture();
        start.mw = tip.position; // what the scripts' Start() would set
        start.hammerSlide = start.potSlide = true;
    }

    // Once the game has finished its own loading (save or intro), take over physics and reset.
    public bool TrySetup()
    {
        if (ready) return true;
        if (!pc.loadFinished) return false;
        pc.enabled = hammer.enabled = potSounds.enabled = craneMove.enabled = false;
        pose.enabled = false; // its per-frame write to Player/Hub's Transform nudges the bodies; Step() runs it instead
        player.GetComponent<Saviour>().enabled = false;  // autosave
        foreach (var j in InLevel<AnchoredJoint2D>()) j.autoConfigureConnectedAnchor = false; // re-created joints keep their anchors
        Physics2D.simulationMode = SimulationMode2D.Script;
        Physics2D.reuseCollisionCallbacks = true;         // no garbage per contact callback; the game never keeps them
        Physics.autoSimulation = false;                   // the game has no 3D bodies
        AudioListener.volume = 0;
        AudioListener.pause = true;
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
        Restore(start);
        return ready = true;
    }

    public void Step(Vector2 action)
    {
        var offset = Vector2.ClampMagnitude(action, 3.5f); // the game keeps its cursor within 3.5 units of the pot
        for (int i = 0; i < K; i++)
        {
            cursor.position = pot.position + offset; // PlayerControl reads the cursor as its target
            hammerFixedUpdate();                     // execution order 0: friction materials from last step's contacts
            potFixedUpdate();
            controlFixedUpdate();                    // execution order 2000: hinge and slider motors toward the cursor
            simTime += dt;
            crane.MovePosition(new Vector2(craneStart.x + Mathf.Sin(simTime) * 0.01f, craneStart.y)); // CraneMove, on sim time
            if (!world.Simulate(dt)) throw new InvalidOperationException("PhysicsScene2D.Simulate did not run");
            if (lake.resetting) lake.CancelInvoke(); // fell in the lake: the flag ends the episode; the game's own reset must not fire
        }
        poseLateUpdate(); // the character's arm pose, once per action in every mode, so rendering never changes the physics
    }

    public int Observe(float[] o)
    {
        // Hammer rates come from body velocities: Unity's joint-speed getters spike to nonsense under manual stepping.
        Vector2 p = pot.position, v = pot.velocity, d = (tip.position - hub.position).normalized;
        o[0] = p.x; o[1] = p.y; o[2] = v.x; o[3] = v.y;
        o[4] = pot.rotation * Mathf.Deg2Rad; o[5] = pot.angularVelocity * Mathf.Deg2Rad;
        o[6] = d.x; o[7] = d.y; o[8] = arm.angularVelocity * Mathf.Deg2Rad;
        o[9] = pc.sj.jointTranslation; o[10] = Vector2.Dot(tip.velocity - hub.velocity, d);
        o[11] = tip.IsTouching(solid) ? 1 : 0;
        o[12] = pot.IsTouching(solid) ? 1 : 0;
        o[13] = lake.resetting ? 1 : 0;
        o[14] = Patches.Won ? 1 : 0;
        for (int i = 0; i < RayCount; i++)
        {
            float a = 2 * Mathf.PI * i / RayCount;
            var hit = world.Raycast(p, new Vector2(Mathf.Cos(a), Mathf.Sin(a)), RayLength, solid);
            o[ObsFields + i] = hit ? hit.distance : RayLength;
        }
        return ObsFields + RayCount;
    }

    public void Reset() => Restore(start);

    public int Save()
    {
        int id = saves.Count;
        saves[id] = Capture();
        return id;
    }

    public bool Load(int id)
    {
        if (!saves.TryGetValue(id, out var s)) return false;
        Restore(s);
        return true;
    }

    public void SetRender(bool on)
    {
        if (cameras == null) cameras = Camera.allCameras; // the level's cameras, all enabled at this point
        foreach (var c in cameras) if (c != null) c.enabled = on;
        Render = on;
    }

    Snapshot Capture()
    {
        int n = bodies.Length;
        var s = new Snapshot
        {
            pos = new Vector2[n], vel = new Vector2[n], rot = new float[n], spin = new float[n], z = new float[n],
            active = new bool[n], awake = new bool[n], type = new RigidbodyType2D[n],
        };
        for (int i = 0; i < n; i++)
        {
            var b = bodies[i];
            s.pos[i] = b.position; s.vel[i] = b.velocity; s.rot[i] = b.rotation; s.spin[i] = b.angularVelocity;
            s.z[i] = b.transform.position.z;
            s.active[i] = b.gameObject.activeSelf; s.awake[i] = b.IsAwake(); s.type[i] = b.bodyType;
        }
        s.gravity = Physics2D.gravity;
        s.simTime = simTime;
        s.hingeSpeed = pc.hj.motor.motorSpeed;
        s.sliderSpeed = pc.sj.motor.motorSpeed;
        s.won = Patches.Won;
        s.mw = (Vector2)F.Mw.GetValue(pc);
        s.mouseSnap = (bool)F.MouseSnap.GetValue(pc);
        s.hammerSlide = (bool)F.HammerSlide.GetValue(hammer);
        s.hammerKeys = Keys(F.HammerPoints.GetValue(hammer));
        s.potSlide = (bool)F.PotSlide.GetValue(potSounds);
        s.potKeys = Keys(F.PotPoints.GetValue(potSounds));
        s.creditsUp = (bool)F.CreditsUp.GetValue(gravity);
        s.giftTime = (float)F.GiftTime.GetValue(giftSpawner);
        s.giftSpawned = (bool)F.GiftSpawned.GetValue(giftSpawner);
        s.giftOpened = (bool)F.GiftOpened.GetValue(gift);
        s.lakeReset = lake.resetting;
        return s;
    }

    void Restore(Snapshot s)
    {
        // Rebuild the level's physics in a brand-new Box2D world, identically every time. Box2D replays bit-exactly only
        // from an identical world: the creation order fixes its broadphase and contact order, and Unity bakes collider
        // shapes and mass properties from the Transforms at creation. Nothing from the current timeline (contacts,
        // warm starts, sleep timers) survives.
        F.HammerSlide.SetValue(hammer, s.hammerSlide); // friction materials first, so re-created contacts start right
        F.PotSlide.SetValue(potSounds, s.potSlide);
        hammerFixedUpdate();
        potFixedUpdate();
        for (int i = 0; i < bodies.Length; i++)
            if (bodies[i].gameObject.activeSelf != s.active[i]) bodies[i].gameObject.SetActive(s.active[i]);
        var active = physicsObjects.Select(g => g.activeSelf).ToArray();
        foreach (var g in physicsObjects) g.SetActive(false);
        // Saved poses before re-creation, parents before children. Depth too: it isn't 2D physics, but Unity bakes the
        // full 3D Transform into shapes and mass properties, and the pose script moves the Hub's depth with the animation.
        for (int i = 0; i < bodies.Length; i++)
            bodies[i].transform.SetPositionAndRotation(
                new Vector3(s.pos[i].x, s.pos[i].y, s.z[i]), Quaternion.Euler(0, 0, s.rot[i]));
        var fresh = SceneManager.CreateScene($"goi-physics-{++generation}", new CreateSceneParameters(LocalPhysicsMode.Physics2D));
        foreach (var root in physicsRoots) SceneManager.MoveGameObjectToScene(root, fresh);
        for (int i = 0; i < physicsObjects.Length; i++) physicsObjects[i].SetActive(active[i]); // fixed creation order
        if (worldScene.IsValid()) SceneManager.UnloadSceneAsync(worldScene);
        worldScene = fresh;
        world = fresh.GetPhysicsScene2D();
        Physics2D.SyncTransforms(); // flush pending Transform changes now, so they can't overwrite the exact states set next
        for (int i = 0; i < bodies.Length; i++)
        {
            var b = bodies[i];
            if (!b.gameObject.activeInHierarchy) continue;
            if (b.bodyType != s.type[i]) b.bodyType = s.type[i];
            b.position = s.pos[i]; b.rotation = s.rot[i]; b.velocity = s.vel[i]; b.angularVelocity = s.spin[i];
            if (s.awake[i]) b.WakeUp(); else b.Sleep();
        }
        var hingeMotor = pc.hj.motor;
        hingeMotor.motorSpeed = s.hingeSpeed;
        pc.hj.motor = hingeMotor;
        var sliderMotor = pc.sj.motor;
        sliderMotor.motorSpeed = s.sliderSpeed;
        pc.sj.motor = sliderMotor;
        // A zero-length step integrates nothing but writes the bodies back to their Transforms (PlayerControl steers
        // from Transform positions) and rebuilds contacts. Its callbacks may touch script state, so restore that after.
        if (!world.Simulate(0f)) throw new InvalidOperationException("PhysicsScene2D.Simulate(0) did not run");
        Physics2D.gravity = s.gravity;
        simTime = s.simTime;
        Patches.Won = s.won;
        F.Mw.SetValue(pc, s.mw);
        F.MouseSnap.SetValue(pc, s.mouseSnap);
        F.PauseInput.SetValue(pc, 0f);
        F.InputsToSkip.SetValue(pc, 0);
        pc.loadFinished = true;
        SetContacts(hammer, F.HammerSlide, F.HammerPoints, F.HammerBuffer, s.hammerSlide, s.hammerKeys);
        SetContacts(potSounds, F.PotSlide, F.PotPoints, F.PotBuffer, s.potSlide, s.potKeys);
        hammerFixedUpdate(); // re-apply the friction materials those flags select
        potFixedUpdate();
        F.CreditsUp.SetValue(gravity, s.creditsUp);
        F.GiftTime.SetValue(giftSpawner, s.giftTime);
        F.GiftSpawned.SetValue(giftSpawner, s.giftSpawned);
        F.GiftOpened.SetValue(gift, s.giftOpened);
        lake.resetting = s.lakeReset;
        lake.CancelInvoke();
        F.FadeIn.SetValue(cam, 0); // clear the lake's fade-to-red
        F.FadeOut.SetValue(cam, 0);
        F.FadeAlpha.SetValue(cam, 0f);
        if (Render) cam.Teleport(pot.position);
    }

    static void SetContacts(object script, FieldInfo slide, FieldInfo points, FieldInfo buffer, bool sliding, Collider2D[] keys)
    {
        slide.SetValue(script, sliding);
        var dict = (Dictionary<Collider2D, ContactPoint2D[]>)points.GetValue(script);
        var shared = (ContactPoint2D[])buffer.GetValue(script);
        dict.Clear();
        foreach (var k in keys) if (k != null) dict[k] = shared; // the game stores its one contact buffer as every value
    }

    static Collider2D[] Keys(object dict) =>
        dict == null ? new Collider2D[0] : ((Dictionary<Collider2D, ContactPoint2D[]>)dict).Keys.ToArray();

    T[] InLevel<T>() where T : Component =>
        Resources.FindObjectsOfTypeAll<T>().Where(x => x.gameObject.scene == player.scene)
            .OrderBy(x => HierarchyKey(x.transform), StringComparer.Ordinal).ToArray();

    // Position in the scene hierarchy (sibling indices from the root): unique and the same in every session, unlike
    // names (the level has same-named siblings, which would fall back to Unity's session-dependent object order).
    static string HierarchyKey(Transform t) =>
        (t.parent == null ? "" : HierarchyKey(t.parent) + "/") + t.GetSiblingIndex().ToString("D5");

    static Action Method(object script, string name) => (Action)Delegate.CreateDelegate(typeof(Action), script,
        script.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic));

    class Snapshot
    {
        public Vector2[] pos, vel;
        public float[] rot, spin, z;
        public bool[] active, awake;
        public RigidbodyType2D[] type;
        public Vector2 gravity, mw;
        public float simTime, giftTime, hingeSpeed, sliderSpeed;
        public bool won, mouseSnap, hammerSlide, potSlide, creditsUp, giftSpawned, giftOpened, lakeReset;
        public Collider2D[] hammerKeys, potKeys;
    }

    // Private game fields that hold physics-relevant state.
    static class F
    {
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        static FieldInfo Of<T>(string name) => typeof(T).GetField(name, Any) ?? throw new MissingFieldException(typeof(T).Name, name);

        public static readonly FieldInfo
            Mw = Of<PlayerControl>("mw"), MouseSnap = Of<PlayerControl>("mouseSnap"),
            PauseInput = Of<PlayerControl>("pauseInputTimer"), InputsToSkip = Of<PlayerControl>("inputsToSkip"),
            HammerSlide = Of<HammerCollisions>("slide"), HammerPoints = Of<HammerCollisions>("collisionPoints"),
            HammerBuffer = Of<HammerCollisions>("contacts"),
            PotSlide = Of<PotSounds>("slide"), PotPoints = Of<PotSounds>("collisionPoints"), PotBuffer = Of<PotSounds>("contacts"),
            CreditsUp = Of<GravityControl>("creditsUp"),
            GiftTime = Of<GiftSpawner>("timeInTrigger"), GiftSpawned = Of<GiftSpawner>("spawned"), GiftOpened = Of<OpenGift>("done"),
            FadeIn = Of<CameraControl>("fadeInTimer"), FadeOut = Of<CameraControl>("fadeOutTimer"), FadeAlpha = Of<CameraControl>("alpha");
    }
}
