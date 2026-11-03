using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using BepInEx.Logging;
using Unity.Collections;
using UnityEngine;

// Lockstep protocol on 127.0.0.1, little-endian. With -goi-connect N the game dials 127.0.0.1:N instead of listening
// (WSL2 forwards Windows' localhost to a listener in WSL, but not the other way round). A game serves one client session and quits when it ends, so a
// crashed or interrupted trainer never leaves games running (and BepInEx allows only 5 per install).
// Request: 1 command byte + fixed-size payload. Response: int32 byte count + payload (float32 obs, or int32s).
// Every obs is followed by one float32 status: 1 while the pot touches the Snake (the chute back to the start).
//   'k' configure(int k, int rays, float rayLength, int render, int width, int height) -> int obs length
//   'r' reset -> obs            's' step(float mouseX, float mouseY) -> obs    'v' save_state -> int id
//   'l' load_state(int id) -> obs, or empty for an unknown id                 'c' close -> empty, then the game quits
//   'd' drop_state(int id) -> empty: frees a saved state
//   'p' place(float along, float dx, float lift, float angle) -> obs, or empty if the climber overlaps the level there
//   'e' dump_engine(int tag) -> empty: the engine's live state to BepInEx/goi-rl-engine-state-<tag>.jsonl (debugging)
//   'n', 'w', 'W', 'g' read the game's native memory: only in a game launched with -goi-native (else they answer empty)
//   'n' dump_native(int tag, int afterStep) -> empty: the native Box2D world to BepInEx/goi-rl-native-<tag>.jsonl, now
//       or (afterStep 1) right after the next physics step's Simulate; a negative tag also searches memory for where
//       the engine settings live (NativeBox2D.ScanSettings: slow, and touches settings)
//   'q' controller -> UTF-8 JSON: the 'controller' row (EngineDump.Controller): deadzone, sensitivity, trackpad, epsilons,
//       the mouse/trackpad/stick AnimationCurves (keys with tangents, weights, wrap modes, plus Unity's own Evaluate on
//       grids), the hammer's and pot's static/sliding PhysicsMaterial2D, PlayerControl's cached motor/slider structs,
//       the scripts' thresholds and buffer sizes, the code literals, a Mono float-evaluation probe, and the names of the
//       recorder's script-state fields
//   'w' record_step(float mouseX, float mouseY) -> one step at K = 1 as Step does it, returned as a binary record
//       (Game.Record, GameRecord.cs): script state before / after the controller / after the step, the native world
//       (bodies, joints, contacts with manifolds and impulses) right before and right after Simulate, and the events
//       between (cursor MovePosition targets, collision callbacks in Unity's order, HammerReturn). Decoder:
//       goi_jax/parity/states.py. Typically 10-40 KB and a few ms.
//   'W' record_action(float mouseX, float mouseY) -> one action at the configured K as Step runs it: int32 K, then K
//       records (int32 bytes + a record as 'w' returns it), one per physics step; the pose update and joint
//       re-creation happen once, after the last step (whose record also holds the Transforms after it, TFM3)
//   'g' geometry -> binary: every fixture of the current world (static pieces included) with its shape, plus the
//       collider/body/joint index tables records refer to (once per session: static pieces never change)
//   'u' route -> UTF-8 JSON (GameRoute.cs): LevelSpline's segments (approximation cache, distances, interpolation
//       points), the lake / Snake / space-zone colliders, ray layers, and probes of the game's own spline queries
//   'o' observe -> obs of the current state without stepping (after a record_step, the obs a step would have returned)
//   'f' frame -> int width, int height, then RGB rows top-down: the image rendered after the last step (render on)
//   'b' props(int on) -> int obs length: Game.PropFields more after the rays (the nearest loose prop), off by default
//   'a' audio -> int sample rate, int channels, then float32 samples: the sound played since the last 'a' (render on)
//   'x' export_state(int id) -> the saved state as bytes (empty for an unknown id), loadable in any other game
//   'i' import_state(int n, then n bytes) -> int id of a new saved state, or -1 if it doesn't fit this level/plugin
class Server
{
    readonly TcpListener listener;
    readonly int connectPort;
    readonly IPAddress[] dialHosts; // tried in turn: 127.0.0.1 (WSL's localhost relay), then -goi-host (the WSL VM itself)
    int dialAttempt;
    float nextDial;
    readonly ManualLogSource log;
    readonly byte[] input = new byte[32];
    readonly byte[] output = new byte[4 + 4 * 256];
    readonly float[] obs = new float[256];
    Socket client;

    public bool FramePending { get; private set; }

    public Server(int port, ManualLogSource log, int connectPort = 0, string host = null)
    {
        this.log = log;
        this.connectPort = connectPort;
        if (connectPort > 0)
        {
            // WSL2's localhost relay sometimes stops forwarding new ports (a launch then never connected in 180 s);
            // the WSL VM's own address doesn't depend on it.
            dialHosts = IPAddress.TryParse(host ?? "", out var extra) ? new[] { IPAddress.Loopback, extra } : new[] { IPAddress.Loopback };
            log.LogInfo($"bridge will dial {string.Join(", ", dialHosts.Select(h => h.ToString()).ToArray())} port {connectPort}");
            return;
        }
        listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        log.LogInfo($"bridge listening on 127.0.0.1:{port}");
    }

    // Reverse mode: one connection attempt per half second until the client's listener answers.
    Socket Dial()
    {
        if (Time.realtimeSinceStartup < nextDial) return null;
        nextDial = Time.realtimeSinceStartup + 0.5f;
        var host = dialHosts[dialAttempt++ % dialHosts.Length];
        var s = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        try
        {
            var pending = s.BeginConnect(host, connectPort, null, null);
            if (pending.AsyncWaitHandle.WaitOne(400) && s.Connected)
            {
                s.EndConnect(pending);
                log.LogInfo($"bridge connected to {host}:{connectPort} (attempt {dialAttempt})");
                return s;
            }
        }
        catch (SocketException) { }
        s.Close();
        return null;
    }

    // Called every frame once the level is ready. Serves commands for up to 50 ms, then hands the frame back to
    // Unity. With rendering on it returns after each step, so every step gets a rendered frame.
    public void Serve(Game game)
    {
        if (client == null)
        {
            if (listener != null)
            {
                if (!listener.Pending()) return;
                client = listener.AcceptSocket();
            }
            else if ((client = Dial()) == null) return;
            client.NoDelay = true;
        }
        if (FramePending) return; // answered by SendFrame once this frame has been rendered
        var deadline = Time.realtimeSinceStartup + 0.05f;
        try
        {
            while (Time.realtimeSinceStartup < deadline)
            {
                if (!client.Poll(1000, SelectMode.SelectRead)) continue;
                if (client.Available == 0) throw new EndOfStreamException("client disconnected");
                Read(1);
                switch ((char)input[0])
                {
                    case 'k':
                        Read(32);
                        game.K = Mathf.Clamp(Int(0), 1, 10000);
                        game.RayCount = Mathf.Clamp(Int(4), 0, obs.Length - Game.ObsFields - 1);
                        game.RayLength = Mathf.Max(Float(8), 0.01f);
                        game.SetRender(Int(12) != 0);
                        if (Int(16) > 0 && Int(20) > 0) Screen.SetResolution(Int(16), Int(20), FullScreenMode.Windowed);
                        game.HeadRayCount = Mathf.Clamp(Int(24), 0, obs.Length - Game.ObsFields - game.RayCount - 1);
                        game.HeadRayLength = Mathf.Max(Float(28), 0.01f);
                        SendInt(game.ObsLength);
                        break;
                    case 'b':
                        Read(4);
                        game.Props = Int(0) != 0;
                        SendInt(game.ObsLength);
                        break;
                    case 'r':
                        game.Reset();
                        SendObs(game);
                        break;
                    case 's':
                        Read(8);
                        game.Step(new Vector2(Float(0), Float(4)));
                        SendObs(game);
                        if (game.Render) return;
                        break;
                    case 'v':
                        SendInt(game.Save());
                        break;
                    case 'l':
                        Read(4);
                        if (game.Load(Int(0))) SendObs(game);
                        else Send(0);
                        break;
                    case 'p':
                        Read(16);
                        if (game.Place(Float(0), Float(4), Float(8), Float(12))) SendObs(game);
                        else Send(0);
                        break;
                    case 'e':
                        Read(4);
                        EngineDump.Write(Path.Combine(BepInEx.Paths.BepInExRootPath, $"goi-rl-engine-state-{Int(0)}.jsonl"), log);
                        Send(0);
                        break;
                    case 'n':
                        Read(8);
                        if (NativeBox2D.Enabled)
                        {
                            var native = Path.Combine(BepInEx.Paths.BepInExRootPath, $"goi-rl-native-{Int(0)}.jsonl");
                            if (Int(4) != 0) NativeBox2D.WriteAfterNextStep(native, log);
                            else NativeBox2D.Write(native, log, Int(0) < 0);
                        }
                        Send(0);
                        break;
                    case 'q':
                        SendBytes(System.Text.Encoding.UTF8.GetBytes(game.ControllerJson()));
                        break;
                    case 'w':
                        Read(8);
                        SendBytes(NativeBox2D.Enabled ? game.Record(new Vector2(Float(0), Float(4))) : new byte[0]);
                        break;
                    case 'W':
                        Read(8);
                        SendBytes(NativeBox2D.Enabled ? game.RecordAction(new Vector2(Float(0), Float(4))) : new byte[0]);
                        break;
                    case 'g':
                        SendBytes(NativeBox2D.Enabled ? game.Geometry() : new byte[0]);
                        break;
                    case 'u':
                        SendBytes(System.Text.Encoding.UTF8.GetBytes(game.RouteJson()));
                        break;
                    case 'o':
                        SendObs(game); // no step: the observation of the current state (Progress's guard moves on as at a step)
                        break;
                    case 'd':
                        Read(4);
                        game.Drop(Int(0));
                        Send(0);
                        break;
                    case 'x':
                        Read(4);
                        SendBytes(game.Export(Int(0)) ?? new byte[0]);
                        break;
                    case 'i':
                        Read(4);
                        var state = new byte[Int(0)];
                        ReadInto(state);
                        SendInt(game.Import(state));
                        break;
                    case 'c':
                        Send(0);
                        Application.Quit();
                        return;
                    case 'f':
                        FramePending = true;
                        return;
                    case 'a':
                        SendBytes(game.TakeAudio());
                        break;
                    default:
                        throw new InvalidDataException($"unknown command byte {input[0]}");
                }
            }
        }
        catch (Exception e)
        {
            EndSession(e);
        }
    }

    // Called at the end of a frame while FramePending, with that frame's image (RGB24): its raw bytes, rows bottom-up as
    // Unity keeps them (the client flips them), sent from one reused buffer. Converting pixel by pixel in managed code
    // and allocating per frame held 1080p filming to ~22 frames a second.
    byte[] frameBytes = new byte[0];
    public void SendFrame(Texture2D frame)
    {
        FramePending = false;
        var px = frame.GetRawTextureData<byte>();
        if (frameBytes.Length != 12 + px.Length) frameBytes = new byte[12 + px.Length];
        Put(frameBytes, 0, 8 + px.Length);
        Put(frameBytes, 4, frame.width);
        Put(frameBytes, 8, frame.height);
        NativeArray<byte>.Copy(px, 0, frameBytes, 12, px.Length);
        try { client.Send(frameBytes); }
        catch (Exception e) { EndSession(e); }
    }

    void EndSession(Exception e)
    {
        log.LogWarning(e is EndOfStreamException || e is SocketException
            ? $"bridge client left: {e.Message}" : $"bridge error: {e}");
        client.Close();
        client = null;
        Application.Quit();
    }

    void Read(int n)
    {
        for (int got = 0; got < n;)
        {
            int r = client.Receive(input, got, n - got, SocketFlags.None);
            if (r == 0) throw new EndOfStreamException("client disconnected");
            got += r;
        }
    }

    void ReadInto(byte[] buffer)
    {
        for (int got = 0; got < buffer.Length;)
        {
            int r = client.Receive(buffer, got, buffer.Length - got, SocketFlags.None);
            if (r == 0) throw new EndOfStreamException("client disconnected");
            got += r;
        }
    }

    void SendBytes(byte[] payload)
    {
        var data = new byte[4 + payload.Length];
        Put(data, 0, payload.Length);
        Buffer.BlockCopy(payload, 0, data, 4, payload.Length);
        client.Send(data);
    }

    int Int(int offset) => BitConverter.ToInt32(input, offset);
    float Float(int offset) => BitConverter.ToSingle(input, offset);

    void SendObs(Game game)
    {
        int n = game.Observe(obs);
        Buffer.BlockCopy(obs, 0, output, 4, n * 4);
        Send(n * 4);
    }

    void SendInt(int v)
    {
        Put(output, 4, v);
        Send(4);
    }

    void Send(int payloadBytes)
    {
        Put(output, 0, payloadBytes);
        client.Send(output, 0, 4 + payloadBytes, SocketFlags.None);
    }

    static void Put(byte[] buffer, int offset, int v)
    {
        buffer[offset] = (byte)v;
        buffer[offset + 1] = (byte)(v >> 8);
        buffer[offset + 2] = (byte)(v >> 16);
        buffer[offset + 3] = (byte)(v >> 24);
    }
}
