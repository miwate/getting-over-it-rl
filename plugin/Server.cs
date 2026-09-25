using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using BepInEx.Logging;
using UnityEngine;

// Lockstep protocol on 127.0.0.1, one client at a time, little-endian.
// Request: 1 command byte + fixed-size payload. Response: int32 byte count + payload (float32 obs, or one int32).
//   'k' configure(int k, int rays, float rayLength, int render, int width, int height) -> int obs length
//   'r' reset -> obs            's' step(float dx, float dy) -> obs            'v' save_state -> int id
//   'l' load_state(int id) -> obs, or empty for an unknown id                 'c' close -> empty, then the game quits
class Server
{
    readonly TcpListener listener;
    readonly ManualLogSource log;
    readonly byte[] input = new byte[32];
    readonly byte[] output = new byte[4 + 4 * 256];
    readonly float[] obs = new float[256];
    Socket client;

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
                        Read(24);
                        game.K = Mathf.Clamp(Int(0), 1, 10000);
                        game.RayCount = Mathf.Clamp(Int(4), 0, obs.Length - Game.ObsFields);
                        game.RayLength = Mathf.Max(Float(8), 0.01f);
                        game.SetRender(Int(12) != 0);
                        if (Int(16) > 0 && Int(20) > 0) Screen.SetResolution(Int(16), Int(20), FullScreenMode.Windowed);
                        SendInt(Game.ObsFields + game.RayCount);
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
                    case 'c':
                        Send(0);
                        Application.Quit();
                        return;
                    default:
                        throw new InvalidDataException($"unknown command byte {input[0]}");
                }
            }
        }
        catch (Exception e)
        {
            log.LogWarning(e is EndOfStreamException || e is SocketException
                ? $"bridge client left: {e.Message}" : $"bridge error, dropping client: {e}");
            client.Close();
            client = null;
        }
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
        Put(4, v);
        Send(4);
    }

    void Send(int payloadBytes)
    {
        Put(0, payloadBytes);
        client.Send(output, 0, 4 + payloadBytes, SocketFlags.None);
    }

    void Put(int offset, int v)
    {
        output[offset] = (byte)v;
        output[offset + 1] = (byte)(v >> 8);
        output[offset + 2] = (byte)(v >> 16);
        output[offset + 3] = (byte)(v >> 24);
    }
}
