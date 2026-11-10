import type { Session } from "../types";

const STORE = "snapshots";
let connection: Promise<IDBDatabase> | undefined;

function openDatabase(): Promise<IDBDatabase> {
  connection ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open("zhixing-agent-history", 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => {
        db.close();
        connection = undefined;
      };
      resolve(db);
    };
  }).catch((error) => {
    connection = undefined;
    throw error;
  });
  return connection;
}

export async function accessSessionSnapshot(
  key: string,
  update?: (stored: Session[] | undefined) => Session[],
): Promise<Session[] | undefined> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE, update ? "readwrite" : "readonly");
    const store = transaction.objectStore(STORE);
    const request = store.get(key);
    let snapshot: Session[] | undefined;
    request.onsuccess = () => {
      snapshot = request.result;
      if (!update) return;
      try {
        // 读取、合并和写回都在同一个事务中，浏览器会串行处理跨标签页写入。
        snapshot = update(snapshot);
        store.put(snapshot, key);
      } catch (error) {
        reject(error);
        transaction.abort();
      }
    };
    transaction.oncomplete = () => resolve(snapshot);
    transaction.onabort = () => reject(transaction.error ?? new Error("会话存储事务已取消。"));
  });
}
