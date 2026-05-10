import { WebSocketServer, WebSocket } from 'ws';
import { IncomingMessage } from 'http';

interface WSClient {
  ws: WebSocket;
  id: string;
  subscribedTopics: Set<string>;
}

export class WebSocketService {
  private wss: WebSocketServer;
  private clients: Map<WebSocket, WSClient> = new Map();

  constructor(wss: WebSocketServer) {
    this.wss = wss;
    this.setupWebSocketServer();
  }

  private setupWebSocketServer(): void {
    this.wss.on('connection', (ws: WebSocket, request: IncomingMessage) => {
      const clientId = this.generateClientId();
      const client: WSClient = {
        ws,
        id: clientId,
        subscribedTopics: new Set(),
      };

      this.clients.set(ws, client);
      console.log(`WebSocket client connected: ${clientId}`);

      ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(client, message);
        } catch (error) {
          console.error('Invalid WebSocket message:', error);
        }
      });

      ws.on('close', () => {
        console.log(`WebSocket client disconnected: ${clientId}`);
        this.clients.delete(ws);
      });

      ws.on('error', (error) => {
        console.error(`WebSocket error for client ${clientId}:`, error);
        this.clients.delete(ws);
      });

      // Send welcome message
      this.sendToClient(ws, {
        type: 'welcome',
        clientId,
        timestamp: new Date().toISOString(),
      });
    });
  }

  private handleMessage(client: WSClient, message: any): void {
    switch (message.type) {
      case 'subscribe':
        if (message.topic) {
          client.subscribedTopics.add(message.topic);
          this.sendToClient(client.ws, {
            type: 'subscribed',
            topic: message.topic,
          });
        }
        break;

      case 'unsubscribe':
        if (message.topic) {
          client.subscribedTopics.delete(message.topic);
          this.sendToClient(client.ws, {
            type: 'unsubscribed',
            topic: message.topic,
          });
        }
        break;

      case 'ping':
        this.sendToClient(client.ws, {
          type: 'pong',
          timestamp: new Date().toISOString(),
        });
        break;

      default:
        console.log(`Unknown message type: ${message.type}`);
    }
  }

  private sendToClient(ws: WebSocket, message: any): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Public methods for broadcasting
  broadcast(topic: string, data: any): void {
    const message = {
      type: 'broadcast',
      topic,
      data,
      timestamp: new Date().toISOString(),
    };

    let sentCount = 0;
    for (const [ws, client] of this.clients) {
      if (client.subscribedTopics.has(topic) || client.subscribedTopics.has('*')) {
        this.sendToClient(ws, message);
        sentCount++;
      }
    }

    if (sentCount > 0) {
      console.log(`Broadcasted to ${sentCount} clients on topic '${topic}'`);
    }
  }

  broadcastToAll(data: any): void {
    const message = {
      type: 'broadcast',
      data,
      timestamp: new Date().toISOString(),
    };

    let sentCount = 0;
    for (const [ws] of this.clients) {
      this.sendToClient(ws, message);
      sentCount++;
    }

    if (sentCount > 0) {
      console.log(`Broadcasted to all ${sentCount} clients`);
    }
  }

  getConnectedClientsCount(): number {
    return this.clients.size;
  }

  isHealthy(): boolean {
    return this.wss !== null && this.wss.address() !== null;
  }
}