using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using BepInEx.Logging;
using UnityEngine;

// Lockstep protocol on 127.0.0.1, little-endian. A game serves one client session and quits when it ends, so a
// crashed or interrupted trainer never leaves games running (and BepInEx allows only 5 per install).
// Request: 1 command byte + fixed-size payload. Response: int32 byte count + payload (float32 obs, or int32s).
// Every obs is followed by one float32 status: 1 while the pot touches the Snake (the chute back to the start).
//   'k' configure(int k, int rays, float rayLength, int render, int width, int height) -> int obs length
//   'r' reset -> obs            's' step(float mouseX, float mouseY) -> obs    'v' save_state -> int id
//   'l' load_state(int id) -> obs, or empty for an unknown id                 'c' close -> empty, then the game quits
//   'd' drop_state(int id) -> empty: frees a saved state
//   'p' place(float along, float dx, float lift, float angle) -> obs, or empty if the climber overlaps the level there
//   'e' dump_engine(int tag) -> empty: the engine's live state to BepInEx/goi-rl-engine-state-<tag>.jsonl (debugging)
//   'f' frame -> int width, int height, then RGB rows top-down: the image rendered after the last step (render on)
//   'x' export_state(int id) -> the saved state as bytes (empty for an unknown id), loadable in any other game
//   'i' import_state(int n, then n bytes) -> int id of a new saved state, or -1 if it doesn't fit this level/plugin
class Server
{
    readonly TcpListener listener;
    readonly ManualLogSource log;
    readonly byte[] input = new byte[32];
    readonly byte[] output = new byte[4 + 4 * 256];
    readonly float[] obs = new float[256];
    Socket client;

    public bool FramePending { get; private set; }

    public Server(int port, ManualLogSource log)
    {
        this.log = log;
        listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        log.LogInfo($"bridge listening on 127.0.0.1:{port}");
    }

    // Called every frame once the level is ready. Serves commands for up to 50 ms, then hands the frame back to
    // Unity. With rendering on it returns after each step, so every step gets a rendered frame.
    public void Serve(Game game)
    {
        if (client == null)
        {
            if (!listener.Pending()) return;
            client = listener.AcceptSocket();
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
                        SendInt(Game.ObsFields + game.RayCount + game.HeadRayCount);
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

    // Called at the end of a frame while FramePending, with that frame's image.
    public void SendFrame(Texture2D frame)
    {
        FramePending = false;
        int w = frame.width, h = frame.height;
        var px = frame.GetPixels32();
        var data = new byte[12 + w * h * 3];
        Put(data, 0, 8 + w * h * 3);
        Put(data, 4, w);
        Put(data, 8, h);
        int o = 12;
        for (int y = h - 1; y >= 0; y--) // Unity's rows run bottom-up
            for (int x = y * w, end = x + w; x < end; x++)
            {
                data[o++] = px[x].r;
                data[o++] = px[x].g;
                data[o++] = px[x].b;
            }
        try { client.Send(data); }
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
