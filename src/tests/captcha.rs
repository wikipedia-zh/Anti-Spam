use super::*;
use crate::captcha::retry_captchas;

fn join_message() -> Message {
    serde_json::from_value(serde_json::json!({
        "message_id":1,"date":0,"chat":{"id":-100,"type":"supergroup","title":"Test"},
        "from":{"id":200,"is_bot":false,"first_name":"New member"}
    }))
    .unwrap()
}

fn reply(text: &str) -> Message {
    let mut message = serde_json::to_value(join_message()).unwrap();
    message["message_id"] = serde_json::json!(2);
    message["text"] = serde_json::json!(text);
    serde_json::from_value(message).unwrap()
}

fn api(failures: Vec<(&str, serde_json::Value)>) -> TelegramStub {
    TelegramStub::with_members(
        vec![],
        failures
            .into_iter()
            .map(|(method, response)| (method.to_string(), -100, response))
            .collect(),
        true,
    )
}

fn api_error() -> serde_json::Value {
    serde_json::json!({"ok":false,"error_code":400,"description":"Bad Request: injected failure"})
}

async fn runtime() -> Arc<Runtime> {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Warn)
        .try_init();
    let runtime = Arc::new(test_runtime().await);
    runtime
        .set_group_module(-100, "captcha", true)
        .await
        .unwrap();
    runtime
}

async fn start(bot: &Bot, runtime: &Arc<Runtime>) {
    let message = join_message();
    start_captcha_challenge(bot, runtime, &message, message.from.as_ref().unwrap()).await;
}

async fn job(runtime: &Runtime) -> Option<serde_json::Value> {
    runtime
        .with_conn(|conn| {
            let mut statement = conn
                .prepare("SELECT payload FROM captcha_jobs WHERE chat_id=-100 AND user_id=200")?;
            let mut rows = statement.query([])?;
            Ok(rows
                .next()?
                .map(|r| serde_json::from_str(&r.get::<_, String>(0).unwrap()).unwrap()))
        })
        .await
        .unwrap()
}

async fn alter_job(runtime: &Runtime, field: &str, value: serde_json::Value) {
    let mut payload = job(runtime).await.unwrap();
    payload[field] = value;
    let raw = payload.to_string();
    runtime
        .with_conn(move |conn| {
            conn.execute(
                "UPDATE captcha_jobs SET payload=?1,next_attempt_at=0",
                params![raw],
            )?;
            Ok(())
        })
        .await
        .unwrap();
}

async fn answer(runtime: &Runtime) -> String {
    let job = job(runtime).await.unwrap();
    (job["a"].as_i64().unwrap() + job["b"].as_i64().unwrap()).to_string()
}

fn calls(telegram: &TelegramStub, method: &str) -> Vec<serde_json::Value> {
    telegram
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(m, _)| m == method)
        .map(|(_, args)| args.clone())
        .collect()
}

#[tokio::test]
async fn accepted_answer_stays_accepted_when_releasing_permissions_fails() {
    let runtime = runtime().await;
    let telegram = api(vec![]);
    start(&telegram.bot, &runtime).await;
    let expected = answer(&runtime).await;
    let failing = api(vec![("restrictchatmember", api_error())]);
    *failing.members.lock().unwrap() = telegram.members.lock().unwrap().clone();
    assert!(check_captcha_and_act(&failing.bot, &runtime, &reply(&expected)).await);
    assert_eq!(job(&runtime).await.unwrap()["state"], "release");
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    alter_job(&restarted, "deadline", serde_json::json!(0)).await;
    retry_captchas(&telegram.bot, &restarted).await.unwrap();
    assert!(job(&restarted).await.is_none());
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 2);
}

#[tokio::test]
async fn finishing_a_kick_keeps_a_subsequent_admin_ban() {
    let runtime = runtime().await;
    let failing = api(vec![("unbanchatmember", api_error())]);
    start(&failing.bot, &runtime).await;
    alter_job(&runtime, "deadline", serde_json::json!(0)).await;
    retry_captchas(&failing.bot, &runtime).await.unwrap();
    assert_eq!(job(&runtime).await.unwrap()["state"], "unkick");
    let telegram = api(vec![]);
    *telegram.members.lock().unwrap() = failing.members.lock().unwrap().clone();
    telegram
        .members
        .lock()
        .unwrap()
        .get_mut(&(-100, 200))
        .unwrap()["until_date"] = serde_json::json!(0);
    alter_job(&runtime, "state", serde_json::json!("unkick")).await;
    retry_captchas(&telegram.bot, &runtime).await.unwrap();
    assert!(job(&runtime).await.is_none());
    assert!(calls(&telegram, "unbanchatmember").is_empty());
}

#[tokio::test]
async fn restart_preserves_answer_and_releases_only_the_captcha_restriction() {
    let runtime = runtime().await;
    let telegram = api(vec![]);
    start(&telegram.bot, &runtime).await;
    assert_eq!(job(&runtime).await.unwrap()["state"], "waiting");
    let expected = answer(&runtime).await;
    let until = calls(&telegram, "restrictchatmember")[0]["until_date"]
        .as_i64()
        .unwrap();
    assert!((150..=180).contains(&(until - Utc::now().timestamp())));
    assert_eq!(
        calls(&telegram, "restrictchatmember")[0]["permissions"]["can_send_messages"],
        true
    );
    let restarted = Arc::new(Runtime::load(runtime.config.clone()).await.unwrap());
    assert!(check_captcha_and_act(&telegram.bot, &restarted, &reply(&expected)).await);
    assert!(job(&restarted).await.is_none());
    let restrictions = calls(&telegram, "restrictchatmember");
    assert_eq!(restrictions.len(), 2);
    assert_eq!(restrictions[1]["permissions"]["can_send_photos"], true);
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert!(!check_captcha_and_act(&telegram.bot, &restarted, &reply(&expected)).await);
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 2);
}

#[tokio::test]
async fn undelivered_question_expires_without_kicking_and_restores_permissions() {
    let runtime = runtime().await;
    let failing = api(vec![("sendmessage", api_error())]);
    start(&failing.bot, &runtime).await;
    assert_eq!(job(&runtime).await.unwrap()["state"], "prepare");
    assert_eq!(calls(&failing, "restrictchatmember").len(), 1);
    alter_job(
        &runtime,
        "deadline",
        serde_json::json!(Utc::now().timestamp() - 1),
    )
    .await;
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let telegram = api(vec![]);
    *telegram.members.lock().unwrap() = failing.members.lock().unwrap().clone();
    assert_eq!(retry_captchas(&telegram.bot, &restarted).await.unwrap(), 1);
    assert!(job(&restarted).await.is_none());
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert_eq!(
        calls(&telegram, "restrictchatmember")[0]["permissions"]["can_send_photos"],
        true
    );
}

#[tokio::test]
async fn expired_challenge_recovers_an_unfinished_kick_after_restart() {
    let runtime = runtime().await;
    let failing = api(vec![("unbanchatmember", api_error())]);
    start(&failing.bot, &runtime).await;
    alter_job(
        &runtime,
        "deadline",
        serde_json::json!(Utc::now().timestamp() - 1),
    )
    .await;
    retry_captchas(&failing.bot, &runtime).await.unwrap();
    assert_eq!(job(&runtime).await.unwrap()["state"], "unkick");
    assert_eq!(calls(&failing, "banchatmember").len(), 1);
    let until = calls(&failing, "banchatmember")[0]["until_date"]
        .as_i64()
        .unwrap();
    assert!((60..=90).contains(&(until - Utc::now().timestamp())));
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    alter_job(&restarted, "state", serde_json::json!("unkick")).await;
    let telegram = api(vec![]);
    *telegram.members.lock().unwrap() = failing.members.lock().unwrap().clone();
    retry_captchas(&telegram.bot, &restarted).await.unwrap();
    assert!(job(&restarted).await.is_none());
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert_eq!(
        calls(&telegram, "unbanchatmember")[0]["only_if_banned"],
        true
    );
}

#[tokio::test]
async fn answer_does_not_clear_a_new_admin_restriction_or_an_independent_ban() {
    for independent in [false, true] {
        let runtime = runtime().await;
        let telegram = api(vec![]);
        start(&telegram.bot, &runtime).await;
        if independent {
            runtime
                .persist_case(&dummy_case(ActionKind::SpamBan, -100, 200, Utc::now()))
                .await
                .unwrap();
        } else {
            let mut members = telegram.members.lock().unwrap();
            let member = members.get_mut(&(-100, 200)).unwrap();
            member["can_send_messages"] = serde_json::json!(false);
            member["until_date"] = serde_json::json!(0);
        }
        let expected = answer(&runtime).await;
        assert!(check_captcha_and_act(&telegram.bot, &runtime, &reply(&expected)).await);
        assert!(job(&runtime).await.is_none());
        assert_eq!(calls(&telegram, "restrictchatmember").len(), 1);
        assert!(calls(&telegram, "unbanchatmember").is_empty());
    }
}

#[tokio::test]
async fn interrupted_kick_does_not_kick_a_rejoined_member_again() {
    let runtime = runtime().await;
    let telegram = api(vec![]);
    start(&telegram.bot, &runtime).await;
    alter_job(
        &runtime,
        "deadline",
        serde_json::json!(Utc::now().timestamp() - 1),
    )
    .await;
    alter_job(&runtime, "kick_started", serde_json::json!(true)).await;
    alter_job(
        &runtime,
        "kick_until",
        serde_json::json!(Utc::now().timestamp() + 90),
    )
    .await;
    alter_job(&runtime, "state", serde_json::json!("kick")).await;
    telegram.members.lock().unwrap().insert(
        (-100, 200),
        serde_json::json!({"status":"member","user":{"id":200,"is_bot":false,"first_name":"Test"}}),
    );
    retry_captchas(&telegram.bot, &runtime).await.unwrap();
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert!(calls(&telegram, "unbanchatmember").is_empty());
    assert!(job(&runtime).await.is_none());
}

#[tokio::test]
async fn duplicate_join_and_concurrent_answers_do_not_duplicate_side_effects() {
    let runtime = runtime().await;
    let telegram = api(vec![]);
    start(&telegram.bot, &runtime).await;
    start(&telegram.bot, &runtime).await;
    assert_eq!(calls(&telegram, "sendmessage").len(), 1);
    let expected = answer(&runtime).await;
    let response = reply(&expected);
    let (first, second, _) = tokio::join!(
        check_captcha_and_act(&telegram.bot, &runtime, &response),
        check_captcha_and_act(&telegram.bot, &runtime, &response),
        retry_captchas(&telegram.bot, &runtime)
    );
    assert!(first || second);
    assert_eq!(calls(&telegram, "restrictchatmember").len(), 2);
    assert!(calls(&telegram, "banchatmember").is_empty());
    assert!(job(&runtime).await.is_none());
}

#[tokio::test]
async fn persistence_failure_does_not_restrict_the_joiner() {
    let runtime = runtime().await;
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_captcha BEFORE INSERT ON captcha_jobs BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    let telegram = api(vec![]);
    start(&telegram.bot, &runtime).await;
    assert!(job(&runtime).await.is_none());
    assert!(telegram.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn question_rate_limit_waits_after_restart_without_losing_the_job() {
    let runtime = runtime().await;
    let limited = api(vec![(
        "sendmessage",
        serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests: retry after 120","parameters":{"retry_after":120}}),
    )]);
    start(&limited.bot, &runtime).await;
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    alter_job(&restarted, "state", serde_json::json!("prepare")).await;
    let telegram = api(vec![]);
    *telegram.members.lock().unwrap() = limited.members.lock().unwrap().clone();
    assert_eq!(retry_captchas(&telegram.bot, &restarted).await.unwrap(), 0);
    assert!(telegram.requests.lock().unwrap().is_empty());
    assert!(job(&restarted).await.is_some());
    restarted
        .with_conn(|conn| {
            conn.execute("UPDATE telegram_retry_state SET not_before=0", [])?;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(retry_captchas(&telegram.bot, &restarted).await.unwrap(), 1);
    assert_eq!(job(&restarted).await.unwrap()["state"], "waiting");
    assert!(
        calls(&telegram, "restrictchatmember").is_empty(),
        "existing restriction is reused"
    );
}

#[tokio::test]
async fn a_new_join_replaces_the_old_deadline_without_lifting_other_restrictions() {
    let runtime = runtime().await;
    let telegram = api(vec![]);
    start(&telegram.bot, &runtime).await;
    let old = job(&runtime).await.unwrap();
    let mut message = join_message();
    message.id = MessageId(3);
    start_captcha_challenge(
        &telegram.bot,
        &runtime,
        &message,
        message.from.as_ref().unwrap(),
    )
    .await;
    let new = job(&runtime).await.unwrap();
    assert_eq!(new["join_message_id"], 3);
    assert_eq!(new["state"], "waiting");
    assert_eq!(new["prior_restrict_until"], old["restrict_until"]);
    assert_eq!(calls(&telegram, "sendmessage").len(), 2);
    assert!(calls(&telegram, "banchatmember").is_empty());
}

#[tokio::test]
async fn operations_pause_releases_old_challenges_even_after_resume() {
    let runtime=runtime().await;let telegram=api(vec![]);start(&telegram.bot,&runtime).await;
    super::operations::set(&runtime,false,true,false).await;
    super::operations::set(&runtime,false,false,false).await;
    let restarted=Runtime::load(runtime.config.clone()).await.unwrap();alter_job(&restarted,"deadline",serde_json::json!(0)).await;
    retry_captchas(&telegram.bot,&restarted).await.unwrap();
    assert!(job(&restarted).await.is_none());assert!(calls(&telegram,"banchatmember").is_empty());assert_eq!(calls(&telegram,"restrictchatmember").len(),2);
}
#[tokio::test]
async fn operations_admission_pause_does_not_interrupt_existing_challenges() {
    let runtime=runtime().await;let telegram=api(vec![]);start(&telegram.bot,&runtime).await;
    super::operations::set(&runtime,true,false,false).await;
    let mut another=serde_json::to_value(join_message()).unwrap();another["message_id"]=serde_json::json!(3);another["from"]["id"]=serde_json::json!(201);let another:Message=serde_json::from_value(another).unwrap();
    start_captcha_challenge(&telegram.bot,&runtime,&another,another.from.as_ref().unwrap()).await;
    assert_eq!(calls(&telegram,"restrictchatmember").len(),1);
    let expected=answer(&runtime).await;check_captcha_and_act(&telegram.bot,&runtime,&reply(&expected)).await;
    assert!(job(&runtime).await.is_none());assert!(calls(&telegram,"banchatmember").is_empty());
}
