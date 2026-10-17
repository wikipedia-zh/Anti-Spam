"""Send an explicitly requested batch without waiting between receiver replies."""
import asyncio
import time
from core import StopBridge
from telegram_client import Flood
from delivery_queue import KEY, ACTIVE, state, pending, changes, rejected, claim, pause, notice


def request(c, actor, request_id):
    if not c.authorized(actor) or not c.config.get('receiver_scope_confirmed'):
        raise StopBridge('Not authorized')
    current = state(c)
    if current.get('mode') == 'burst' and current.get('status') in ACTIVE:
        return '批次已啟動；用 /queue 查看進度。'
    waiting = pending(c)
    if any(op['state'] != 'sent' and not rejected(op) for op in waiting):
        return '有操作的送出結果不明，不能重複發送；請先核對。'
    todo = changes(c)
    if len(todo) > 200:
        return '單批超過 200 筆，請分批核對。'
    if any(rejected(op) and (op['action'], op['user_id'], op['case_id']) not in todo for op in waiting):
        return '失敗案件已不再符合收錄條件，請先核對。'
    if not todo and not waiting:
        return '沒有待寫入項目。'
    c.ledger.put(KEY, {'status': 'running', 'mode': 'burst', 'id': request_id,
                      'actor': actor, 'items': [list(x) for x in todo],
                      'retry_ids': [x['id'] for x in waiting if rejected(x)],
                      'operation': None, 'checked_at': 0})
    return '開始連續送出已核准項目，不逐筆等回覆；結果會另行核對，Telegram 限流時會等待。'


async def advance(c, t):
    run = state(c)
    if not c.authorized(run['actor']) or not c.config.get('receiver_scope_confirmed'):
        pause(c, '啟動者或接收端權限已變更。')
        return
    waiting = pending(c)
    if waiting and int(time.time()) - run.get('checked_at', 0) >= 5:
        await t.reconcile_batch(c.ledger, waiting)
        run['checked_at'] = int(time.time())
        with c.db:
            c.ledger.put(KEY, run)
    # Bound each pass so commands, notices and membership checks keep running.
    for _ in range(10):
        waiting = pending(c)
        if any(op['state'] != 'sent' and op['id'] not in run['retry_ids'] for op in waiting):
            pause(c, 'hexDrill 回覆失敗或有送出結果未確認的操作，已停止剩餘發送。')
            return
        available = set(changes(c))
        outstanding = {(op['action'], op['user_id']) for op in waiting if not rejected(op)}
        batch_users = {x[1] for x in run['items']}
        removals = sorted(x for x in available if x[0] == 'unban' and x[1] in batch_users)
        targets = removals + [tuple(x) for x in run['items']]
        todo = [x for x in targets if x in available and x[:2] not in outstanding]
        if not todo:
            if waiting:
                return
            run['status'] = 'done'
            with c.db:
                c.ledger.put(KEY, run)
                c.ledger.put('delivery_paused', False)
                notice(c, run, 'done', '✅ 本批 +spb 寫入及回覆核對已完成。')
            return
        change = todo[0]
        previous = [op for op in waiting if (op['action'], op['user_id']) == change[:2] and rejected(op)]
        for op in previous:
            if op['id'] not in run['retry_ids'] or not await t.confirmed_failure(op):
                pause(c, '無法核對原失敗回覆，未重送。')
                return
        if not c.authorized(run['actor']) or change not in changes(c):
            pause(c, '權限或來源案件已變更。')
            return
        operation = claim(c, run, change, previous)
        if operation is None:
            return
        try:
            message = await asyncio.wait_for(t.send(operation), timeout=25)
            c.ledger.sent(operation['id'], message)
        except Flood as error:
            c.ledger.flood(operation['id'], error.seconds)
            raise
        except BaseException:
            c.ledger.uncertain(operation['id'])
            pause(c, '送出結果不明，已停止剩餘發送，未重送。')
            raise
