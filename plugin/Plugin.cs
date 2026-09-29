using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using BepInEx;
using BepInEx.Logging;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;

[BepInPlugin("goi-rl.bridge", "GOI RL Bridge", "0.2.0")]
public class Plugin : BaseUnityPlugin
{
    // Launch flags. Without them the game plays normally apart from the 1 Hz log line.
    // -goi-rl: RL mode. Skips the menu, serves the bridge on 127.0.0.1 (-goi-port N, default 5555), steps physics
    //          only on request, and never writes PlayerPrefs. -goi-dump: write engine-only data once per level load.
    // -goi-colliders: draw the physics shapes over the game (see ColliderOverlay).
    static readonly string[] Args = Environment.GetCommandLineArgs();
    readonly bool rlMode = Args.Contains("-goi-rl");
    readonly bool dump = Args.Contains("-goi-dump");

    Rigidbody2D pot;
    float nextLog, dumpAt = float.MaxValue;
    bool menuSkipped;
    Server server;
    Game game;

    void Awake()
    {
        if (Args.Contains("-goi-colliders")) ColliderOverlay.Enable();
        if (!rlMode) return;
        Application.runInBackground = true; // off in the game's player settings, so unfocused instances would stall
        Patches.Log = Logger;
        var harmony = Harmony.CreateAndPatchAll(typeof(Patches));
        harmony.PatchAll(typeof(VirtualMouse));
        int patched = harmony.GetPatchedMethods().Count();
        Logger.LogInfo($"RL mode: {patched} methods patched");
        server = new Server(IntArg("-goi-port", 5555), Logger);
        StartCoroutine(AnswerFrameRequests());
        SceneManager.sceneLoaded += (scene, _) => { if (scene.name == "Mian") game = new Game(); }; // before Start(): authored start pose
    }

    // A frame can only be read once rendering has finished, so frame requests are answered at the end of the frame.
    IEnumerator AnswerFrameRequests()
    {
        var endOfFrame = new WaitForEndOfFrame();
        while (true)
        {
            yield return endOfFrame;
            if (!server.FramePending) continue;
            var frame = ScreenCapture.CaptureScreenshotAsTexture();
            server.SendFrame(frame);
            Destroy(frame);
        }
    }

    void Update()
    {
        if (rlMode)
        {
            if (!menuSkipped) SkipMenu();
            Cursor.lockState = CursorLockMode.None; // the game captures the mouse; a background RL instance shouldn't
            Cursor.visible = true;
            if (game != null && game.TrySetup()) server.Serve(game);
        }
        if (pot == null)
        {
            var player = GameObject.Find("Player");
            if (player == null) return;
            pot = player.GetComponent<Rigidbody2D>();
            dumpAt = Time.unscaledTime + 1f; // let every Start() run first
        }
        if (dump && Time.unscaledTime >= dumpAt)
        {
            dumpAt = float.MaxValue;
            EngineDump.Write(Path.Combine(Paths.BepInExRootPath, "goi-rl-engine-dump.jsonl"), Logger);
        }
        if (Time.unscaledTime < nextLog) return;
        nextLog = Time.unscaledTime + 1f;
        var p = pot.position;
        Logger.LogInfo($"pot position=({p.x:F2}, {p.y:F2}) height={p.y:F2}");
    }

    void SkipMenu()
    {
        var loader = FindObjectOfType<Loader>();
        if (loader == null) return;
        var ready = (bool)typeof(Loader).GetField("loadFinished", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(loader);
        if (!ready) return;
        loader.DoStart(); // releases the pending "Mian" activation; unlike StartGame it doesn't delete the save
        menuSkipped = true;
    }

    static int IntArg(string name, int fallback)
    {
        int i = Array.IndexOf(Args, name);
        return i >= 0 && i + 1 < Args.Length && int.TryParse(Args[i + 1], out var v) ? v : fallback;
    }
}

// Installed in RL mode only. The game's controller reads the mouse through Rewired; the agent's virtual mouse replaces
// it (and the gamepad sticks read 0), so a real mouse can't reach RL games either.
[HarmonyPatch]
static class VirtualMouse
{
    public static Vector2 Input;

    static MethodBase TargetMethod() => AccessTools.Method("Rewired.Player:GetAxis", new[] { typeof(string) });

    static bool Prefix(string actionName, ref float __result)
    {
        __result = actionName == "mouseX" ? Input.x : actionName == "mouseY" ? Input.y : 0f;
        return false;
    }
}

// Installed in RL mode only.
[HarmonyPatch]
static class Patches
{
    public static bool Won;
    public static ManualLogSource Log;
    static readonly HashSet<string> blocked = new HashSet<string>();

    // RL runs must never touch the player's save or settings (PlayerPrefs = the Windows registry). DeleteKey and
    // Save are native and can't be patched; the one deletion a level can trigger is the win, handled below.
    [HarmonyPrefix, HarmonyPatch(typeof(PlayerPrefs), nameof(PlayerPrefs.SetInt))]
    static bool SetInt(string key) => Block(key);

    [HarmonyPrefix, HarmonyPatch(typeof(PlayerPrefs), nameof(PlayerPrefs.SetFloat))]
    static bool SetFloat(string key) => Block(key);

    [HarmonyPrefix, HarmonyPatch(typeof(PlayerPrefs), nameof(PlayerPrefs.SetString))]
    static bool SetString(string key) => Block(key);

    static bool Block(string key)
    {
        if (blocked.Add(key)) Log.LogInfo($"blocked PlayerPrefs write: {key}");
        return false;
    }

    // The chapel's bat flock and the gift's are visual (the Bat prefab has no collider or body), but once they have
    // flown, BatGen and OpenGift destroy their own objects, which have colliders (the gift a body too). A restore can't
    // rebuild a destroyed physics object: the next load failed and the game quit. In RL mode they don't start.
    [HarmonyPrefix, HarmonyPatch(typeof(BatGen), "OnTriggerEnter2D")]
    static bool NoBats() => false;

    [HarmonyPrefix, HarmonyPatch(typeof(OpenGift), "OnCollisionEnter2D")]
    static bool NoGiftBats() => false;

    // Leaving the space zone through the top rolls the credits and deletes the save. In RL mode it only ends the episode.
    [HarmonyPrefix, HarmonyPatch(typeof(GravityControl), "OnTriggerExit2D")]
    static bool Win(GravityControl __instance, Collider2D coll)
    {
        var rb = coll.attachedRigidbody;
        if (rb == null || rb.position.y <= __instance.GetComponent<BoxCollider2D>().bounds.max.y - 5f) return true;
        Won = true;
        return false;
    }
}
