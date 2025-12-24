import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertContractAnalysisSchema } from "@shared/schema";
import crypto from "crypto";
import dotenv from "dotenv";

// Загружаем переменные окружения из .env файла
dotenv.config();

const DEFAULT_DEEPSEEK_TIMEOUT_MS = Number(process.env.DEEPSEEK_TIMEOUT_MS ?? 60_000);
const DEFAULT_DEEPSEEK_RETRIES = Number(process.env.DEEPSEEK_RETRY_COUNT ?? 3);

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function parseApiKeys(): string[] {
  const apiKeyEnv = process.env.VITE_DEEPSEEK_API_KEYS || process.env.VITE_API_KEY;
  return (
    apiKeyEnv
      ?.split(",")
      .map((k) => k.trim())
      .filter(Boolean) ?? []
  );
}

function buildChatUrl(rawBaseUrl: string): string {
  return rawBaseUrl.endsWith("/chat/completions")
    ? rawBaseUrl
    : `${rawBaseUrl.replace(/\/?$/, "")}/chat/completions`;
}

export async function registerRoutes(app: Express): Promise<Server> {
  // API endpoint для получения ключа API DeepSeek
  app.get("/api/deepseek-key", (req, res) => {
    const apiKey = process.env.VITE_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ message: "DeepSeek API key is not configured" });
    }
    res.json({ apiKey });
  });

  // Базовый URL можно переопределить через DEEPSEEK_API_URL
  // Допускается как полный путь /chat/completions, так и базовый https://api.artemox.com/v1
  const rawBaseUrl = process.env.DEEPSEEK_API_URL || "https://api.artemox.com/v1";
  const DEEPSEEK_API_URL = buildChatUrl(rawBaseUrl);

  // Прокси для вызовов DeepSeek, чтобы обходить CORS и не раскрывать ключ в браузере
  app.post("/api/deepseek/chat", async (req, res) => {
    const apiKeys = parseApiKeys();

    if (!apiKeys.length) {
      return res.status(500).json({ message: "DeepSeek API key is not configured" });
    }

    const maxAttempts = Math.max(1, DEFAULT_DEEPSEEK_RETRIES);
    const timeoutMs = Math.max(1_000, DEFAULT_DEEPSEEK_TIMEOUT_MS);
    let lastError: unknown;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const apiKey = apiKeys[attempt % apiKeys.length];
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(new Error("DeepSeek request timed out")),
        timeoutMs
      );

      try {
        const upstreamResponse = await fetch(DEEPSEEK_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify(req.body),
          signal: controller.signal,
        });

        const payload = await upstreamResponse
          .json()
          .catch(() => null as unknown as Record<string, unknown> | null);

        const canRetryUpstreamError =
          upstreamResponse.status >= 500 && upstreamResponse.status < 600 && attempt < maxAttempts - 1;

        if (!upstreamResponse.ok && canRetryUpstreamError) {
          lastError = payload ?? upstreamResponse.statusText;
          await sleep(300 * (attempt + 1));
          continue;
        }

        if (!upstreamResponse.ok) {
          return res.status(upstreamResponse.status).json(
            payload ?? {
              message: "Failed to call DeepSeek API",
              status: upstreamResponse.status,
            }
          );
        }

        return res.json(payload);
      } catch (error) {
        lastError = error;
        if (attempt < maxAttempts - 1) {
          await sleep(300 * (attempt + 1));
          continue;
        }

        console.error("DeepSeek proxy error:", error);
        return res.status(500).json({
          message: "Failed to call DeepSeek API",
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        clearTimeout(timeoutId);
      }
    }

    console.error("DeepSeek proxy error:", lastError);
    return res.status(500).json({
      message: "Failed to call DeepSeek API",
      error: lastError instanceof Error ? lastError.message : String(lastError),
    });
  });

  // Store contract analysis result
  app.post("/api/analysis", async (req, res) => {
    try {
      const { contractText, analysisResult } = req.body;
      
      if (!contractText || !analysisResult) {
        return res.status(400).json({ message: "Contract text and analysis result are required" });
      }

      // Создаем хеш для договора
      const contractHash = crypto.createHash('sha256').update(contractText).digest('hex');
      
      const analysisData = {
        contractHash,
        analysisResult,
        createdAt: new Date().toISOString(),
      };

      const validatedData = insertContractAnalysisSchema.parse(analysisData);
      const result = await storage.createContractAnalysis(validatedData);
      
      res.json({ success: true, id: result.id, hash: contractHash });
    } catch (error) {
      console.error("Error storing analysis:", error);
      res.status(500).json({ message: "Failed to store analysis" });
    }
  });

  // Get analysis by contract hash
  app.get("/api/analysis/:hash", async (req, res) => {
    try {
      const { hash } = req.params;
      
      if (!hash) {
        return res.status(400).json({ message: "Contract hash is required" });
      }

      const analysis = await storage.getAnalysisByHash(hash);
      
      if (!analysis) {
        return res.status(404).json({ message: "Analysis not found" });
      }

      res.json(analysis);
    } catch (error) {
      console.error("Error retrieving analysis:", error);
      res.status(500).json({ message: "Failed to retrieve analysis" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
