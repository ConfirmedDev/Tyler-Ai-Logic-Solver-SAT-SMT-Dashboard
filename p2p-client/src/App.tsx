import { useRef, useState } from "react";

// use your actual Render URLs here
const SIGNALING_URL = "wss://tyler-p2p-signaling.onrender.com"; 
const FLASK_URL = "https://tyler-ai-logic-solver-sat-smt-dashboard.onrender.com";


type CollatzState = {
  roomId: string;
  clausesText: string;
  collatzLogs: string[];
  reducedClauses: string[];
  twoSatLogs: string[];
  twoSatResult: "SAT" | "UNSAT" | null;
  reducerTime: number | null;
  twoSatTime: number | null;
};

const initialState: CollatzState = {
  roomId: "",
  clausesText: "",
  collatzLogs: [],
  reducedClauses: [],
  twoSatLogs: [],
  twoSatResult: null,
  reducerTime: null,
  twoSatTime: null,
};

function App() {
  const [state, setState] = useState<CollatzState>(initialState);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");

  const wsRef = useRef<WebSocket | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);

  // --- helper to send full state to peer ---
  const sendStateToPeer = (newState: CollatzState) => {
    const dc = dataChannelRef.current;
    if (dc && dc.readyState === "open") {
      dc.send(JSON.stringify({ type: "SYNC_STATE", state: newState }));
    }
  };

  // --- WebRTC setup helpers ---
  const createPeerConnection = () => {
    if (pcRef.current) return pcRef.current;

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    pc.onicecandidate = (event) => {
      if (event.candidate && wsRef.current && state.roomId) {
        wsRef.current.send(
          JSON.stringify({
            type: "ice-candidate",
            roomId: state.roomId,
            payload: event.candidate,
          })
        );
      }
    };

    pc.ondatachannel = (event) => {
      const channel = event.channel;
      dataChannelRef.current = channel;
      channel.onopen = () => {
        setConnectionStatus("connected");
      };
      channel.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "SYNC_STATE") {
            setState(msg.state);
          }
        } catch {
          // ignore
        }
      };
    };

    pcRef.current = pc;
    return pc;
  };

  const createOffer = async () => {
    const pc = createPeerConnection();
    // if we are offerer, we create data channel
    const dc = pc.createDataChannel("collatz");
    dataChannelRef.current = dc;

    dc.onopen = () => {
      setConnectionStatus("connected");
    };
    dc.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "SYNC_STATE") {
          setState(msg.state);
        }
      } catch {
        // ignore
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    if (wsRef.current && state.roomId) {
      wsRef.current.send(
        JSON.stringify({
          type: "offer",
          roomId: state.roomId,
          payload: offer,
        })
      );
    }
  };

  const handleOffer = async (offer: RTCSessionDescriptionInit) => {
    const pc = createPeerConnection();
    await pc.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    if (wsRef.current && state.roomId) {
      wsRef.current.send(
        JSON.stringify({
          type: "answer",
          roomId: state.roomId,
          payload: answer,
        })
      );
    }
  };

  const handleAnswer = async (answer: RTCSessionDescriptionInit) => {
    const pc = createPeerConnection();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  };

  const handleIceCandidate = async (candidate: RTCIceCandidateInit) => {
    const pc = createPeerConnection();
    try {
      await pc.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (e) {
      console.error("Error adding ICE candidate", e);
    }
  };

  // --- Signaling WebSocket setup ---
  const connectToRoom = () => {
    if (!state.roomId) {
      alert("Enter a room ID first.");
      return;
    }
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(SIGNALING_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus("waiting for peer...");
      ws.send(
        JSON.stringify({
          type: "join",
          roomId: state.roomId,
        })
      );
    };

    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data);
      if (!msg) return;

      switch (msg.type) {
        case "peer-joined":
          // we are the existing peer, create an offer
          createOffer();
          break;
        case "offer":
          await handleOffer(msg.payload);
          break;
        case "answer":
          await handleAnswer(msg.payload);
          break;
        case "ice-candidate":
          await handleIceCandidate(msg.payload);
          break;
        default:
          break;
      }
    };

    ws.onclose = () => {
      setConnectionStatus("disconnected");
      pcRef.current?.close();
      pcRef.current = null;
      dataChannelRef.current = null;
    };
  };

  // --- update clauses + broadcast ---
  const updateClausesText = (text: string) => {
    const newState = { ...state, clausesText: text };
    setState(newState);
    sendStateToPeer(newState);
  };

  // --- call your existing Flask endpoints ---
  const runCollatzPipeline = async () => {
    if (!state.clausesText.trim()) {
      alert("Enter some 3-SAT clauses first.");
      return;
    }

    try {
      // Step 1: Collatz clause reducer
      const clauses = state.clausesText
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

      const res1 = await fetch(`${FLASK_URL}/collatz_clause_reducer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clauses }),
      });
      if (!res1.ok) {
        throw new Error(await res1.text());
      }
      const data1 = await res1.json();

      const reduced = data1.reduced_clauses as string[];
      const res2 = await fetch(`${FLASK_URL}/collatz_2sat_reducer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clauses: reduced }),
      });
      if (!res2.ok) {
        throw new Error(await res2.text());
      }
      const data2 = await res2.json();

      const newState: CollatzState = {
        ...state,
        collatzLogs: data1.logs,
        reducedClauses: reduced,
        reducerTime: data1.reducer_time,
        twoSatLogs: data2.logs,
        twoSatResult: data2.result,
        twoSatTime: data2.twosat_time,
      };

      setState(newState);
      sendStateToPeer(newState);
    } catch (e: any) {
      console.error(e);
      alert("Error running Collatz pipeline: " + e.message);
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", padding: "16px" }}>
      <h1>Tyler&apos;s P2P Collatz SAT Lab</h1>

      <div
        style={{
          display: "flex",
          gap: "1rem",
          marginBottom: "1rem",
          alignItems: "center",
        }}
      >
        <input
          type="text"
          placeholder="Room ID"
          value={state.roomId}
          onChange={(e) =>
            setState((prev) => ({ ...prev, roomId: e.target.value }))
          }
          style={{ padding: "4px 8px" }}
        />
        <button onClick={connectToRoom}>Connect</button>
        <span>Status: {connectionStatus}</span>
      </div>

      <div style={{ display: "flex", gap: "1rem" }}>
        <div style={{ flex: 1 }}>
          <h2>Shared 3-SAT Clauses</h2>
          <textarea
            rows={18}
            style={{ width: "100%", fontFamily: "monospace" }}
            value={state.clausesText}
            onChange={(e) => updateClausesText(e.target.value)}
            placeholder="!x2 !x14 !x8&#10;x4 !x16 x7&#10;..."
          />
          <button style={{ marginTop: "8px" }} onClick={runCollatzPipeline}>
            Run Collatz → 2-SAT → Check
          </button>
        </div>

        <div style={{ flex: 1 }}>
          <h3>Collatz Reducer Logs</h3>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              maxHeight: "150px",
              overflow: "auto",
            }}
          >
            {state.collatzLogs.join("\n")}
          </pre>

          <h3>Reduced 2-SAT Clauses</h3>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              maxHeight: "120px",
              overflow: "auto",
            }}
          >
            {state.reducedClauses.join("\n")}
          </pre>

          <h3>2-SAT Logs</h3>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              maxHeight: "150px",
              overflow: "auto",
            }}
          >
            {state.twoSatLogs.join("\n")}
          </pre>

          <h3>Result</h3>
          <pre>
            {state.twoSatResult
              ? (state.twoSatResult === "SAT" ? "✅ SAT" : "❌ UNSAT") +
                ` (Reducer: ${state.reducerTime}s, 2-SAT: ${state.twoSatTime}s)`
              : "No run yet."}
          </pre>
        </div>
      </div>
    </div>
  );
}

export default App;

