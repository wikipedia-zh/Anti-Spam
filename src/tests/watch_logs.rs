use super::*;
use crate::watch_logs::{decide, load, valid_case};

const REQUEST: &str = "aab3bf04-0000-4000-8000-000000000000";
const CASE: &str = "d7f68ca9-0000-4000-8000-000000000000";

fn databases(title: &str, action: &str, status: &str) -> (Connection, Connection) {
    let source = Connection::open_in_memory().unwrap();
    source.execute_batch("CREATE TABLE cases(id TEXT,action TEXT,status TEXT,target_user_id INTEGER,chat_id INTEGER,netban_eligible INTEGER);
        CREATE TABLE origin_ban_jobs(case_id TEXT,ban_done INTEGER,outcome_unknown INTEGER);
        CREATE TABLE global_whitelist(user_id INTEGER);CREATE TABLE maintainers(user_id INTEGER);CREATE TABLE reviewers(user_id INTEGER);").unwrap();
    source
        .execute(
            "INSERT INTO cases VALUES (?1,?2,?3,8803865976,-1003518168907,1)",
            params![CASE, action, status],
        )
        .unwrap();
    let bridge = Connection::open_in_memory().unwrap();
    bridge
        .execute_batch(include_str!("../../bridge/log_import.sql"))
        .unwrap();
    bridge.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);CREATE TABLE sudo_users(user_id INTEGER);
        INSERT INTO meta VALUES ('binding','{\"account_id\":11,\"hex_bot_id\":12}'),('watch_owner','500'),
        ('spy_settings','{\"destination\":-10001,\"excluded\":[],\"bot_id\":13}');INSERT INTO sudo_users VALUES (600);").unwrap();
    bridge
        .execute(
            "INSERT INTO meta VALUES ('watch_group_verified_at',?1)",
            [Utc::now().timestamp().to_string()],
        )
        .unwrap();
    bridge.execute("INSERT INTO watch_log_requests(id,case_id,user_id,source_chat,title,submitted_by,submitted_message,created_at,updated_at,review_state,review_chat,review_message)
        VALUES (?1,?2,8803865976,-1003518168907,?3,500,1,?4,?4,'sent',-10001,10)",params![REQUEST,CASE,title,Utc::now().timestamp()]).unwrap();
    (source, bridge)
}

#[test]
fn existing_ban_logs_require_owner_confirmation_and_do_not_change_spb() {
    for (title, action, status) in [
        ("訪客模式機器人封禁", "guest_bot_ban", "guest_bot_banned"),
        ("自動封禁", "auto_ban", "auto_banned"),
        ("受理封禁", "report_approved", "approved_and_banned"),
        ("逕行封禁", "spam_ban", "force_approved"),
    ] {
        let (source, mut bridge) = databases(title, action, status);
        let before = source.total_changes();
        assert!(valid_case(&source, &bridge, &load(&bridge, REQUEST).unwrap().unwrap()).unwrap());
        assert_eq!(load(&bridge, REQUEST).unwrap().unwrap().status, "pending");
        for (actor, chat, message) in [
            (600, -10001, 10),
            (777, -10001, 10),
            (500, 500, 10),
            (500, -10002, 10),
            (500, -10001, 11),
        ] {
            assert!(decide(&source, &mut bridge, REQUEST, actor, chat, message, true).is_err());
        }
        assert_eq!(load(&bridge, REQUEST).unwrap().unwrap().status, "pending");
        assert!(decide(&source, &mut bridge, REQUEST, 500, -10001, 10, true)
            .unwrap()
            .starts_with("已確認"));
        assert_eq!(load(&bridge, REQUEST).unwrap().unwrap().status, "approved");
        let actor: i64 = bridge
            .query_row("SELECT approved_by FROM watch_log_requests", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(actor, 500);
        let changes = bridge.total_changes();
        assert!(decide(&source, &mut bridge, REQUEST, 500, -10001, 10, true)
            .unwrap()
            .contains("未重複"));
        assert!(
            decide(&source, &mut bridge, REQUEST, 500, -10001, 10, false)
                .unwrap()
                .contains("未重複")
        );
        assert_eq!(bridge.total_changes(), changes);
        assert_eq!(source.total_changes(), before);
    }
}

#[test]
fn cancel_expiry_revocation_and_protection_never_approve() {
    for sql in [
        "UPDATE cases SET status='reversal_pending'",
        "UPDATE cases SET netban_eligible=0",
        "UPDATE cases SET target_user_id=123",
        "UPDATE cases SET chat_id=-10002",
        "UPDATE cases SET action='auto_ban'",
        "INSERT INTO global_whitelist VALUES (8803865976)",
        "INSERT INTO maintainers VALUES (8803865976)",
        "INSERT INTO reviewers VALUES (8803865976)",
        "INSERT INTO origin_ban_jobs SELECT id,0,1 FROM cases",
    ] {
        let (source, mut bridge) =
            databases("訪客模式機器人封禁", "guest_bot_ban", "guest_bot_banned");
        source.execute(sql, []).unwrap();
        assert!(decide(&source, &mut bridge, REQUEST, 500, -10001, 10, true)
            .unwrap()
            .contains("未推送"));
        assert_eq!(load(&bridge, REQUEST).unwrap().unwrap().status, "invalid");
    }
    for sql in [
        "UPDATE watch_log_requests SET created_at=0",
        "INSERT INTO sudo_users VALUES (8803865976)",
    ] {
        let (source, mut bridge) =
            databases("訪客模式機器人封禁", "guest_bot_ban", "guest_bot_banned");
        bridge.execute(sql, []).unwrap();
        assert!(decide(&source, &mut bridge, REQUEST, 500, -10001, 10, true)
            .unwrap()
            .contains("未推送"));
    }
    let (source, mut bridge) = databases("訪客模式機器人封禁", "guest_bot_ban", "guest_bot_banned");
    assert!(
        decide(&source, &mut bridge, REQUEST, 500, -10001, 10, false)
            .unwrap()
            .starts_with("已取消")
    );
    assert_eq!(load(&bridge, REQUEST).unwrap().unwrap().status, "rejected");
}

#[test]
fn lost_card_delivery_and_changed_group_verification_fail_closed() {
    for sql in [
        "UPDATE watch_log_requests SET review_state='sending'",
        "UPDATE watch_log_requests SET review_state='uncertain'",
        "UPDATE watch_log_requests SET review_message=NULL",
        "UPDATE watch_log_requests SET submitted_by=600",
        "UPDATE meta SET value='0' WHERE key='watch_group_verified_at'",
    ] {
        let (source, mut bridge) =
            databases("訪客模式機器人封禁", "guest_bot_ban", "guest_bot_banned");
        bridge.execute(sql, []).unwrap();
        assert!(decide(&source, &mut bridge, REQUEST, 500, -10001, 10, true).is_err());
        assert_eq!(load(&bridge, REQUEST).unwrap().unwrap().status, "pending");
    }
}
