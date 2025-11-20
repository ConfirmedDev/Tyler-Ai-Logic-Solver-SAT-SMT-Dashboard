import express from "express";
import http from "http";
import { WebSocketServer } from "ws";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json({ status: "ok", message: "P2P signaling server running" });
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// roomId -> Set of WebSocket clients
const rooms = new Map();

function joinRoom(roomId, ws) {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, new Set());
  }
  rooms.get(roomId).add(ws);
  ws.roomId = roomId;
}

function leaveRoom(ws) {
  const roomId = ws.roomId;
  if (!roomId) return;
  const room = rooms.get(roomId);
  if (!room) return;
  room.delete(ws);
  if (room.size === 0) {
    rooms.delete(roomId);
  }
}

wss.on("connection", (ws) => {
  ws.on("message", (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      console.error("Invalid JSON from client");
      return;
    }

    const { type, roomId, payload } = msg;

    if (type === "join") {
      joinRoom(roomId, ws);
      // notify existing peers that someone joined (so they can create an offer)
      const room = rooms.get(roomId);
      if (!room) return;
      for (const client of room) {
        if (client !== ws && client.readyState === client.OPEN) {
          client.send(JSON.stringify({ type: "peer-joined", roomId }));
        }
      }
      return;
    }

    // Forward WebRTC signaling messages (offer/answer/ICE)
    if (["offer", "answer", "ice-candidate"].includes(type)) {
      const room = rooms.get(roomId);
      if (!room) return;
      for (const client of room) {
        if (client !== ws && client.readyState === client.OPEN) {
          client.send(JSON.stringify({ type, roomId, payload }));
        }
      }
    }
  });

  ws.on("close", () => {
    leaveRoom(ws);
  });
});

const PORT = process.env.PORT || 4000;
server.listen(PORT, () => {
  console.log(`Signaling server listening on http://localhost:${PORT}`);
});
