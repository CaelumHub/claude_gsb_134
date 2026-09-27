"""协作聊天: 按天分片持久化 + REST 拉取(WS 实时推送在 ws.py)。"""
from __future__ import annotations

import asyncio
import os
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from . import auth, config
from .boards import board_ctx, manager
from .models import ChatPostReq
from .storage import JsonlLog, now_ms

router = APIRouter(prefix="/api/boards", tags=["chat"])

_chat_logs: Dict[str, JsonlLog] = {}
_chat_locks: Dict[str, asyncio.Lock] = {}


def _log_for(board_id: str) -> JsonlLog:
    log = _chat_logs.get(board_id)
    if log is None:
        directory = os.path.join(config.board_dir(board_id), "chat")
        log = JsonlLog(directory, prefix="chat", shard_fmt="%Y%m%d")   # 按天分片
        _chat_logs[board_id] = log
    return log


def _lock_for(board_id: str) -> asyncio.Lock:
    lock = _chat_locks.get(board_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[board_id] = lock
    return lock


async def append_message(board_id: str, user: Dict[str, Any], text: str,
                         kind: str = "msg") -> Dict[str, Any]:
    """持久化一条聊天消息(JSONL 追加, 按天分片, 原子性见 storage.py)。"""
    settings = config.get_settings()
    max_len = int(settings.get("chat_message_max_len") or 2000)
    message = {
        "id": "m" + secrets.token_hex(8),
        "board_id": board_id,
        "user": user.get("username"),
        "display_name": user.get("display_name") or user.get("username"),
        "color": user.get("color") or "#5b8ff9",
        "text": (text or "")[:max_len],
        "kind": kind if kind in ("msg", "system") else "msg",
        "ts": now_ms(),
    }
    async with _lock_for(board_id):
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: _log_for(board_id).append([message], ts_ms=message["ts"]))
    return message


def _msg_key(m: Dict[str, Any]) -> Tuple[int, str]:
    """消息的时间序键: ts 在前, 同毫秒用 id 兜底(随机 id 不影响时间序)。"""
    return (m.get("ts") or 0, m.get("id") or "")


def load_messages(board_id: str, before_ts: Optional[int] = None,
                  before_id: Optional[str] = None,
                  limit: int = 100) -> Tuple[List[Dict[str, Any]], bool]:
    """倒序游标分页(返回前翻正序): 游标 (before_ts, before_id) 之前的 limit 条。

    必须跨分片(跨天)连续扫描, 直到收集满 limit+1 条「合格」消息才能停止,
    而不能按读到的原始记录数提前 break —— 当天消息很多时, 新分片里绝大
    多数记录都会被游标过滤掉, 按原始计数停止会永远翻不到更早日期的分片。
    分片内记录不保证按时间有序(乱序写入/补写历史消息), 因此每个分片必须
    整体读完再判定: 更老的分片整天都更早, 不可能挤进当前页, 提前停止才安全。

    游标用 (ts, id) 复合键: 同一毫秒有多条消息时, 只用 ts 做游标会漏掉
    同 ts 中 id 更小的消息(或在下一页重复返回)。before_id 缺省时退化为
    严格的 ts < before_ts, 兼容旧调用方。
    返回 (messages, has_more), has_more 表示游标之前仍有消息。
    """
    log = _log_for(board_id)
    shards = log.list_shards()
    if before_ts is not None:
        # 只扫可能包含更早消息的分片(分片名=日期)
        day = datetime.fromtimestamp(before_ts / 1000.0).strftime("%Y%m%d")
        shards = [s for s in shards if s[len("chat-"): -len(".jsonl")] <= day]

    def qualifies(m: Dict[str, Any]) -> bool:
        if before_ts is None:
            return True
        key = _msg_key(m)
        if before_id:
            return key < (before_ts, before_id)
        return (m.get("ts") or 0) < before_ts

    want = limit + 1                       # 多取一条判定 has_more
    picked: List[Dict[str, Any]] = []
    for name in reversed(shards):
        picked.extend(m for m in log.read_shard(name) if qualifies(m))
        if len(picked) >= want:
            break

    has_more = len(picked) > limit
    picked.sort(key=_msg_key)
    return picked[-limit:], has_more


@router.get("/{board_id}/chat")
async def get_chat(board_id: str,
                   before: Optional[int] = Query(default=None),
                   before_id: Optional[str] = Query(default=None),
                   limit: int = Query(default=100, ge=1, le=500),
                   user: Dict[str, Any] = Depends(auth.current_user)):
    await board_ctx(board_id, user, "viewer")
    loop = asyncio.get_running_loop()
    messages, has_more = await loop.run_in_executor(
        None, load_messages, board_id, before, before_id, limit)
    return {"messages": messages, "has_more": has_more}


@router.post("/{board_id}/chat")
async def post_chat(board_id: str, req: ChatPostReq,
                    user: Dict[str, Any] = Depends(auth.current_user)):
    await board_ctx(board_id, user, "viewer")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    message = await append_message(board_id, user, req.text.strip(), req.kind)
    # REST 发送的消息也推给在线 WS 客户端
    from .ws import conn_manager
    await conn_manager.broadcast(board_id, {"type": "chat", "message": message},
                                 exclude_client=None)
    return {"message": message}


def system_note(board_id: str, text: str) -> Dict[str, Any]:
    """构造一条本地系统消息(不落盘, 仅广播, 如 加入/离开)。"""
    return {
        "id": "m" + secrets.token_hex(8),
        "board_id": board_id,
        "user": "system",
        "display_name": "系统",
        "color": "#8a93a5",
        "text": text,
        "kind": "system",
        "ts": now_ms(),
    }
