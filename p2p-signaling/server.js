const WebSocket = require("ws");
const http = require("http");

const PORT = process.env.PORT || 4000;

// create an HTTP server (Render expects an HTTP service)
const server = http.createServer();
const wss = new WebSocket.Server({ server });

const rooms = new Map();

wss.on("connection", (ws) => {
  ws.on("message", (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }

    const { type, roomId, payload } = msg;
    if (!roomId) return;

    if (type === "join") {
      if (!rooms.has(roomId)) rooms.set(roomId, []);
      rooms.get(roomId).push(ws);

      rooms.get(roomId).forEach((client) => {
        if (client !== ws && client.readyState === WebSocket.OPEN) {
          client.send(JSON.stringify({ type: "peer-joined", roomId }));
        }
      });
    } else if (["offer", "answer", "ice-candidate"].includes(type)) {
      const peers = rooms.get(roomId) || [];
      peers.forEach((client) => {
        if (client !== ws && client.readyState === WebSocket.OPEN) {
          client.send(JSON.stringify({ type, roomId, payload }));
        }
      });
    }
  });

  ws.on("close", () => {
    for (const [roomId, peers] of rooms.entries()) {
      rooms.set(
        roomId,
        peers.filter((p) => p !== ws)
      );
      if (rooms.get(roomId).length === 0) rooms.delete(roomId);
    }
  });
});

server.listen(PORT, () => {
  console.log(`Signaling server listening on port ${PORT}`);
});
