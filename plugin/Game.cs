using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using FluffyUnderware.Curvy;
using UnityEngine;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

// Lockstep control of the loaded level. The game's own controller and friction scripts still do the work: they are
// disabled so Unity stops calling their FixedUpdate, and Step() calls it instead, in the game's script execution
// order, then simulates one physics step of the level's own physics world (see Restore). Unity still sends collision
// callbacks to disabled scripts, so their contact bookkeeping keeps working. Replays from a snapshot are bit-exact.
class Game
{
    // Route points observed, in distance along the route past the pot (AHEAD in goi_env/client.py): dense close by, sparser
    // far out. Linesight, the best Trackmania AI, sees 40 points of track ahead.
    static readonly float[] Ahead = { 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50 };
    public static readonly int ObsFields = 23 + 2 * Ahead.Length; // layout: OBS_FIELDS in goi_env/client.py, then rays
    public int K = 4;                // physics steps per action
    public int RayCount = 16;
    public float RayLength = 10f;
    public int HeadRayCount;            // rays from the hammer head, after the pot's: what it can hook (0: none)
    public float HeadRayLength = 4f;
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
    readonly CurvySpline route; // the level route ProgressMeter measures progress along
    readonly Teleporter[] teleporters; // the space zone's two sides, which wrap the climber sideways
    readonly Collider2D snake;         // a frictionless chute from beside the bucket down to the start
    readonly Rigidbody2D pot, hub, arm, tip, cursor, crane;
    readonly Vector2 craneStart;
    readonly Rigidbody2D[] bodies;       // every body in the level, in a fixed order
    readonly HingeJoint2D[] hinges;      // every hinge in the level, parents before children
    readonly int[][] hingeBodies;        // per hinge: its body, its connected body, then every body hanging from that
    readonly GameObject[] physicsRoots;  // every root holding 2D physics; moved into a fresh physics world on restore
    readonly GameObject[] physicsObjects; // top-most objects with 2D physics, in path order: re-created in this order
    readonly Action hammerFixedUpdate, potFixedUpdate, controlFixedUpdate, poseLateUpdate;
    readonly PoseControl pose;
    readonly ContactFilter2D solid;
    readonly Snapshot start;
    readonly Dictionary<int, Snapshot> saves = new Dictionary<int, Snapshot>(); // ~1 KB each, until dropped
    readonly Collider2D[] colliders; // every collider in the level, in a fixed order: contact keys of exported states
    // The snapshot the world was last rebuilt from, until the next step: a save then returns it. A capture of a rebuilt
    // world isn't bit-identical to what it was rebuilt from (the hammer Tip's out-of-plane pose round-trips through its
    // Transform, and its baked collider, inertia and centre of mass move in the last digits), so rebuilding twice
    // differed from rebuilding once: a backtrack in a search, then a save and load, and the path no longer replayed.
    Snapshot restored;
    readonly Collider2D[] overlap = new Collider2D[1];
    int nextSave;
    Camera[] cameras;
    Scene worldScene;
    PhysicsScene2D world;
    int generation;
    float simTime;
    float progress;  // route progress at the last observation (Progress)
    const float MaxAdvance = 3f; // route units per observation that always count (the pot moves ~1 per step at most)
    const float OnRoute = 3f;    // a pot this close to the route point is there, however far ahead it is
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
        route = Object.FindObjectOfType<ProgressMeter>().GetComponent<CurvySpline>();
        teleporters = InLevel<Teleporter>();
        snake = GameObject.Find("Snake").GetComponent<Collider2D>();
        craneStart = crane.position;
        bodies = InLevel<Rigidbody2D>();
        hinges = InLevel<HingeJoint2D>();
        hingeBodies = hinges.Select(h => new[] { Array.IndexOf(bodies, h.attachedRigidbody), Array.IndexOf(bodies, h.connectedBody) }
            .Concat(Enumerable.Range(0, bodies.Length).Where(i => h.connectedBody != null && bodies[i] != h.connectedBody
                && bodies[i].transform.IsChildOf(h.connectedBody.transform))).ToArray()).ToArray();
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
        colliders = InLevel<Collider2D>().OrderBy(c => HierarchyKey(c.transform), StringComparer.Ordinal)
            .ThenBy(c => Array.IndexOf(c.GetComponents<Collider2D>(), c)).ToArray(); // the same in every launch
        start = Capture();
        start.mw = tip.position; // what the scripts' Start() would set
        start.hammerSlide = start.potSlide = true;
        start.progress = route.TFToDistance(route.GetNearestPointTF(player.transform.position));
        for (int i = 0; i < teleporters.Length; i++) start.teleport[3 * i + 2] = true; // enableTeleport
    }

    // Once the game has finished its own loading (save or intro), take over physics and reset.
    public bool TrySetup()
    {
        if (ready) return true;
        if (!pc.loadFinished) return false;
        pc.enabled = hammer.enabled = potSounds.enabled = craneMove.enabled = false;
        foreach (var t in teleporters) t.enabled = false; // Step() wraps the climber; their triggers still report
        pose.enabled = false; // its per-frame write to Player/Hub's Transform nudges the bodies; Step() runs it instead
        pc.trackpad = false;  // the game applies the player's own mouse settings; RL games use the mouse defaults
        F.Sensitivity.SetValue(pc, 1f);
        pose.anim.enabled = false; // visual only: its own per-frame update would undo the pose script's hands-on-handle IK
        // A restore deactivates and re-activates the Player (to re-create its bodies), which reset the character's
        // animators: the opening pose's layer blended back in and the mesh deformed more with every load (videos of
        // search paths, with a load every half second). Visual only: the pose never reaches the physics (Step).
        pose.anim.keepAnimatorControllerStateOnDisable = true;
        pose.potAnim.keepAnimatorControllerStateOnDisable = true;
        cam.enabled = false;       // visual only: a camera spring on Unity's clock trails a game that runs faster than real time
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

    // action: the mouse input the game's controller reads every physics step, as if a player moved the mouse. The game's
    // own acceleration curve, smoothing, 3.5 reach and idle drift all apply: a still mouse lets the cursor settle on
    // the hammer head, and the hammer goes loose.
    public void Step(Vector2 action)
    {
        restored = null;
        VirtualMouse.Input = Vector2.ClampMagnitude(action, 100f);
        for (int i = 0; i < K; i++)
        {
            hammerFixedUpdate();                     // execution order 0: friction materials from last step's contacts
            potFixedUpdate();
            controlFixedUpdate();                    // execution order 2000: moves the cursor by the mouse, motors toward it
            simTime += dt;
            crane.MovePosition(new Vector2(craneStart.x + Mathf.Sin(simTime) * 0.01f, craneStart.y)); // CraneMove, on sim time
            if (!world.Simulate(dt)) throw new InvalidOperationException("PhysicsScene2D.Simulate did not run");
            if (lake.resetting) lake.CancelInvoke(); // fell in the lake: the flag ends the episode; the game's own reset must not fire
            foreach (var t in teleporters) Teleport(t);
            if (Render) cam.FixedUpdate();           // the camera follows on sim time
        }
        // The character's arm pose, once per action in every mode, so rendering never changes the physics. It also puts the
        // hammer at the depth of the animated right hand, which follows frame time, and a moved Transform re-syncs the
        // hammer's bodies: keep their depth fixed, so the physics can't see the animation.
        var hubAt = hub.transform.position;
        poseLateUpdate();
        hub.transform.position = hubAt;
    }

    // What a Teleporter's LateUpdate does, on sim time right after the physics step whose trigger asked for it (the game
    // does it on the frame clock, and moves Camera.main, which doesn't exist with rendering off): the climber and the
    // cursor jump to the other side's x, and that side ignores the arrival until the climber leaves it.
    void Teleport(Teleporter t)
    {
        if (t.shouldEnable)
        {
            t.enableTeleport = true;
            t.shouldEnable = false;
        }
        if (!t.shouldTeleport) return;
        t.targetX.GetComponent<Teleporter>().enableTeleport = false;
        var shift = new Vector3(t.targetX.position.x - player.transform.position.x, 0, 0);
        player.transform.position += shift;
        t.cursor.position += shift;
        Physics2D.SyncTransforms();
        if (Render) cam.Teleport(pot.position);
        t.shouldTeleport = false;
        t.enableTeleport = false;
        t.shouldEnable = true;
    }

    // Route progress as ProgressMeter measures it (the distance along LevelSpline to its point nearest the pot), except
    // that it can't leap ahead. Where the route doubles back, the nearest point jumps: under the box pile past the
    // playground it read 325 for a pot near 290, and exploration and rewards chased such jumps in most sections. A
    // forward leap counts only as far as the distance to the route keeps shrinking along it from the last value (a real
    // advance does; a flip to another part of the route behind a barrier doesn't), unless the pot is right at the new
    // point (hanging under the girder's end is genuine; under the boxes the pot was ~10 from its "325"). Falling back
    // counts at once.
    float Progress(Vector3 position)
    {
        float Gap(float s) => Vector2.Distance(route.Interpolate(route.DistanceToTF(s)), position);
        float nearest = route.TFToDistance(route.GetNearestPointTF(position));
        if (nearest <= progress + MaxAdvance || Gap(nearest) <= OnRoute) return progress = nearest;
        float best = progress, bestGap = Gap(progress);
        for (float s = progress + 0.5f; s <= nearest; s += 0.5f)
        {
            float gap = Gap(s);
            if (gap > bestGap + 0.05f) break; // moving on along the route takes us away from the pot again
            if (gap <= bestGap)
            {
                best = s;
                bestGap = gap;
            }
        }
        return progress = best;
    }

    // The observation, then one status value that isn't part of it: 1 while the pot touches the Snake.
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
        float along = Progress(player.transform.position);
        o[15] = along;
        Vector2 c = cursor.position, head = tip.position; // the cursor is where the controller aims next
        o[16] = c.x - p.x; o[17] = c.y - p.y;
        o[18] = c.x - head.x; o[19] = c.y - head.y;
        var mouse = (Vector2)F.MouseInput.GetValue(pc); // the controller's smoothed mouse input
        o[20] = mouse.x; o[21] = mouse.y;
        o[22] = (float)F.MouseSpeed.GetValue(pc);      // its running mouse speed: the cursor's gain, and no idle drift when fast
        for (int i = 0; i < Ahead.Length; i++) // where the route goes next, relative to the pot
        {
            Vector2 q = route.Interpolate(route.DistanceToTF(along + Ahead[i])); // LevelSpline sits at the origin, unrotated
            o[23 + 2 * i] = q.x - p.x;
            o[24 + 2 * i] = q.y - p.y;
        }
        for (int i = 0; i < RayCount; i++)
        {
            float a = 2 * Mathf.PI * i / RayCount;
            var hit = world.Raycast(p, new Vector2(Mathf.Cos(a), Mathf.Sin(a)), RayLength, solid);
            o[ObsFields + i] = hit ? hit.distance : RayLength;
        }
        for (int i = 0; i < HeadRayCount; i++)
        {
            float a = 2 * Mathf.PI * i / HeadRayCount;
            var hit = world.Raycast(tip.position, new Vector2(Mathf.Cos(a), Mathf.Sin(a)), HeadRayLength, solid);
            o[ObsFields + RayCount + i] = hit ? hit.distance : HeadRayLength;
        }
        o[ObsFields + RayCount + HeadRayCount] = pot.IsTouching(snake) ? 1 : 0;
        return ObsFields + RayCount + HeadRayCount + 1;
    }

    public void Reset() => Restore(start);

    public int Save()
    {
        int id = nextSave++;
        saves[id] = restored ?? Capture();
        return id;
    }

    public void Drop(int id) => saves.Remove(id);

    // A saved state as bytes, and back in another game (another launch): a state found in one game can start play in
    // any other, without replaying the moves that led there (long replays drift across launches). Bodies, hinges,
    // teleporters and colliders are in hierarchy order, the same in every launch; contact keys travel as collider indices.
    const int StateFormat = 1;

    public byte[] Export(int id)
    {
        if (!saves.TryGetValue(id, out var s)) return null;
        using (var m = new MemoryStream())
        using (var w = new BinaryWriter(m))
        {
            foreach (var n in new[] { StateFormat, bodies.Length, hinges.Length, teleporters.Length, colliders.Length }) w.Write(n);
            void V(Vector2 v) { w.Write(v.x); w.Write(v.y); }
            for (int i = 0; i < bodies.Length; i++)
            {
                V(s.pos[i]); V(s.vel[i]); w.Write(s.rot[i]); w.Write(s.spin[i]); w.Write(s.z[i]);
                w.Write(s.pose[i].x); w.Write(s.pose[i].y); w.Write(s.pose[i].z); w.Write(s.pose[i].w);
                w.Write(s.active[i]); w.Write(s.awake[i]); w.Write((int)s.type[i]);
            }
            V(s.gravity); V(s.mw); V(s.mouseInput); V(s.oldMouse);
            foreach (var f in new[] { s.simTime, s.giftTime, s.hingeSpeed, s.sliderSpeed, s.mouseSpeed, s.progress }) w.Write(f);
            foreach (var b in new[] { s.won, s.mouseSnap, s.hammerSlide, s.potSlide, s.creditsUp, s.giftSpawned, s.giftOpened,
                                      s.lakeReset }) w.Write(b);
            foreach (var b in s.teleport) w.Write(b);
            foreach (var a in s.hingeAngle) w.Write(a);
            foreach (var keys in new[] { s.hammerKeys, s.potKeys })
            {
                w.Write(keys.Length);
                foreach (var k in keys) w.Write(Array.IndexOf(colliders, k));
            }
            return m.ToArray();
        }
    }

    // An exported state as a new save id, or -1 if it came from a different level layout or plugin version.
    public int Import(byte[] data)
    {
        using (var r = new BinaryReader(new MemoryStream(data)))
        {
            foreach (var n in new[] { StateFormat, bodies.Length, hinges.Length, teleporters.Length, colliders.Length })
                if (r.ReadInt32() != n) return -1;
            Vector2 V() => new Vector2(r.ReadSingle(), r.ReadSingle());
            int count = bodies.Length;
            var s = new Snapshot
            {
                pos = new Vector2[count], vel = new Vector2[count], rot = new float[count], spin = new float[count],
                z = new float[count], pose = new Quaternion[count], active = new bool[count], awake = new bool[count],
                type = new RigidbodyType2D[count],
            };
            for (int i = 0; i < count; i++)
            {
                s.pos[i] = V(); s.vel[i] = V(); s.rot[i] = r.ReadSingle(); s.spin[i] = r.ReadSingle(); s.z[i] = r.ReadSingle();
                s.pose[i] = new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                s.active[i] = r.ReadBoolean(); s.awake[i] = r.ReadBoolean(); s.type[i] = (RigidbodyType2D)r.ReadInt32();
            }
            s.gravity = V(); s.mw = V(); s.mouseInput = V(); s.oldMouse = V();
            s.simTime = r.ReadSingle(); s.giftTime = r.ReadSingle(); s.hingeSpeed = r.ReadSingle();
            s.sliderSpeed = r.ReadSingle(); s.mouseSpeed = r.ReadSingle(); s.progress = r.ReadSingle();
            s.won = r.ReadBoolean(); s.mouseSnap = r.ReadBoolean(); s.hammerSlide = r.ReadBoolean(); s.potSlide = r.ReadBoolean();
            s.creditsUp = r.ReadBoolean(); s.giftSpawned = r.ReadBoolean(); s.giftOpened = r.ReadBoolean(); s.lakeReset = r.ReadBoolean();
            s.teleport = Enumerable.Range(0, 3 * teleporters.Length).Select(_ => r.ReadBoolean()).ToArray();
            s.hingeAngle = Enumerable.Range(0, hinges.Length).Select(_ => r.ReadSingle()).ToArray();
            Collider2D[] Keys() => Enumerable.Range(0, r.ReadInt32()).Select(_ => r.ReadInt32())
                .Select(k => k >= 0 && k < colliders.Length ? colliders[k] : null).Where(c => c != null).ToArray();
            s.hammerKeys = Keys();
            s.potKeys = Keys();
            int id = nextSave++;
            saves[id] = s;
            return id;
        }
    }

    // Start somewhere along the route instead of the bottom: the start state with the pot at the route point `along`
    // moved by (dx, lift), and the hammer swung about the hub to point at `angle` degrees, the cursor on its head. The
    // rest of the level stays as at the start. False if the climber overlaps the level there: place again, or reset.
    public bool Place(float along, float dx, float lift, float angle)
    {
        var s = start.Copy();
        int Index(Rigidbody2D b) => Array.IndexOf(bodies, b);
        Vector2 shift = (Vector2)route.Interpolate(route.DistanceToTF(along)) + new Vector2(dx, lift) - s.pos[Index(pot)];
        Vector2 hubAt = s.pos[Index(hub)] + shift;
        float turn = angle - Vector2.SignedAngle(Vector2.right, s.pos[Index(tip)] - s.pos[Index(hub)]);
        var spin = Quaternion.Euler(0, 0, turn);
        s.hingeAngle = (float[])s.hingeAngle.Clone();
        s.progress = along;
        s.hingeAngle[Array.IndexOf(hinges, pc.hj)] += turn; // the hammer's hinge turns with it
        for (int i = 0; i < bodies.Length; i++)
        {
            var t = bodies[i].transform;
            if (!t.IsChildOf(player.transform)) continue;
            s.pos[i] += shift;
            if (!t.IsChildOf(arm.transform)) continue; // the hammer: the hub hinge's free end and everything under it
            s.pos[i] = hubAt + (Vector2)(spin * (Vector3)(s.pos[i] - hubAt));
            s.rot[i] += turn;
            s.pose[i] = spin * s.pose[i];
        }
        s.pos[Index(cursor)] = s.mw = s.pos[Index(tip)];
        Restore(s);
        foreach (var c in player.GetComponentsInChildren<Collider2D>())
            if (c.enabled && !c.isTrigger && c.OverlapCollider(solid, overlap) > 0) return false;
        return true;
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
            pose = new Quaternion[n],
            active = new bool[n], awake = new bool[n], type = new RigidbodyType2D[n],
        };
        for (int i = 0; i < n; i++)
        {
            var b = bodies[i];
            s.pos[i] = b.position; s.vel[i] = b.velocity; s.rot[i] = b.rotation; s.spin[i] = b.angularVelocity;
            s.z[i] = b.transform.position.z;
            s.pose[i] = b.transform.rotation;
            s.active[i] = b.gameObject.activeSelf; s.awake[i] = b.IsAwake(); s.type[i] = b.bodyType;
        }
        s.gravity = Physics2D.gravity;
        s.simTime = simTime;
        s.hingeSpeed = pc.hj.motor.motorSpeed;
        s.sliderSpeed = pc.sj.motor.motorSpeed;
        s.won = Patches.Won;
        s.mw = (Vector2)F.Mw.GetValue(pc);
        s.mouseInput = (Vector2)F.MouseInput.GetValue(pc);
        s.oldMouse = (Vector2)F.OldMouse.GetValue(pc);
        s.mouseSpeed = (float)F.MouseSpeed.GetValue(pc);
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
        s.teleport = teleporters.SelectMany(t => new[] { t.shouldTeleport, t.shouldEnable, t.enableTeleport }).ToArray();
        s.hingeAngle = hinges.Select(h => h.jointAngle).ToArray();
        s.progress = progress;
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
        // A hinge re-created on re-activation keeps its joint angle and re-derives its reference angle from its bodies'
        // Transform angles, which are wrapped to +-180: the reference then depends on the angle it had before, and a
        // hammer spun a full turn in the last run shifts it by 360 (the controller's motor math subtracts it, so it
        // rounds differently, and a limited hinge would jump). So first turn each hinge's hanging part, as one piece,
        // until the angle it carries over re-creates the saved joint angle once the saved rotations are set.
        for (int k = 0; k < hinges.Length; k++)
        {
            var h = hinges[k];
            int[] ib = hingeBodies[k];
            if (!h.isActiveAndEnabled || ib[1] < 0) continue;
            float a = s.rot[ib[0]], b = s.rot[ib[1]];
            float carried = Wrap(b) - Wrap(a) - (b - a - s.hingeAngle[k]);
            float turn = carried - h.jointAngle;
            if (turn != 0) foreach (int i in ib.Skip(1)) bodies[i].rotation += turn;
        }
        var active = physicsObjects.Select(g => g.activeSelf).ToArray();
        foreach (var g in physicsObjects) g.SetActive(false);
        // Saved poses before re-creation, parents before children. Depth too: it isn't 2D physics, but Unity bakes the
        // full 3D Transform into shapes and mass properties, and the pose script moves the Hub's depth with the animation.
        // The full 3D rotation too: the hammer's Handle and Tip are authored tilted out of the screen plane (the two
        // tilts cancel at the Tip), and the hammer mesh and grip points hang under them.
        for (int i = 0; i < bodies.Length; i++)
            bodies[i].transform.SetPositionAndRotation(new Vector3(s.pos[i].x, s.pos[i].y, s.z[i]), s.pose[i]);
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
        progress = s.progress;
        Patches.Won = s.won;
        F.Mw.SetValue(pc, s.mw);
        F.MouseInput.SetValue(pc, s.mouseInput);
        F.OldMouse.SetValue(pc, s.oldMouse);
        F.MouseSpeed.SetValue(pc, s.mouseSpeed);
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
        for (int i = 0; i < teleporters.Length; i++)
        {
            teleporters[i].shouldTeleport = s.teleport[3 * i];
            teleporters[i].shouldEnable = s.teleport[3 * i + 1];
            teleporters[i].enableTeleport = s.teleport[3 * i + 2];
        }
        F.FadeIn.SetValue(cam, 0); // clear the lake's fade-to-red
        F.FadeOut.SetValue(cam, 0);
        F.FadeAlpha.SetValue(cam, 0f);
        // The camera jumps only with the climber (a reset, a placement, a backtrack): a load where it already is, like
        // the rebuilds at a replay's cuts, would cut the video.
        if (Render && Vector2.Distance(cam.transform.position, pot.position) > 5f) cam.Teleport(pot.position);
        restored = s;
    }

    static float Wrap(float degrees) => degrees - 360f * Mathf.Round(degrees / 360f);

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
        public Quaternion[] pose; // Transform rotation; rot is the physics angle
        public bool[] active, awake;
        public RigidbodyType2D[] type;
        public Vector2 gravity, mw, mouseInput, oldMouse;
        public float simTime, giftTime, hingeSpeed, sliderSpeed, mouseSpeed;
        public bool won, mouseSnap, hammerSlide, potSlide, creditsUp, giftSpawned, giftOpened, lakeReset;
        public bool[] teleport; // per teleporter: shouldTeleport, shouldEnable, enableTeleport
        public float[] hingeAngle; // per hinge: its joint angle
        public float progress;     // route progress at capture (continuity of Progress)
        public Collider2D[] hammerKeys, potKeys;

        public Snapshot Copy()
        {
            var c = (Snapshot)MemberwiseClone();
            c.pos = (Vector2[])pos.Clone(); c.rot = (float[])rot.Clone(); c.pose = (Quaternion[])pose.Clone();
            return c;
        }
    }

    // Private game fields that hold physics-relevant state.
    static class F
    {
        const BindingFlags Any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        static FieldInfo Of<T>(string name) => typeof(T).GetField(name, Any) ?? throw new MissingFieldException(typeof(T).Name, name);

        public static readonly FieldInfo
            Mw = Of<PlayerControl>("mw"), MouseSnap = Of<PlayerControl>("mouseSnap"),
            PauseInput = Of<PlayerControl>("pauseInputTimer"), InputsToSkip = Of<PlayerControl>("inputsToSkip"),
            MouseInput = Of<PlayerControl>("mouseInput"), OldMouse = Of<PlayerControl>("oldMouse"),
            MouseSpeed = Of<PlayerControl>("mouseVelocityAverage"), Sensitivity = Of<PlayerControl>("mouseSensitivity"),
            HammerSlide = Of<HammerCollisions>("slide"), HammerPoints = Of<HammerCollisions>("collisionPoints"),
            HammerBuffer = Of<HammerCollisions>("contacts"),
            PotSlide = Of<PotSounds>("slide"), PotPoints = Of<PotSounds>("collisionPoints"), PotBuffer = Of<PotSounds>("contacts"),
            CreditsUp = Of<GravityControl>("creditsUp"),
            GiftTime = Of<GiftSpawner>("timeInTrigger"), GiftSpawned = Of<GiftSpawner>("spawned"), GiftOpened = Of<OpenGift>("done"),
            FadeIn = Of<CameraControl>("fadeInTimer"), FadeOut = Of<CameraControl>("fadeOutTimer"), FadeAlpha = Of<CameraControl>("alpha");
    }
}
