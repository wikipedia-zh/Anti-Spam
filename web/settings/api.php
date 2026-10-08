<?php
declare(strict_types=1);
header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');
header('Referrer-Policy: no-referrer');
header('X-Content-Type-Options: nosniff');

function fail(int $status, string $code): void {
    http_response_code($status);
    echo json_encode(['error' => $code]);
    exit;
}

$origin = getenv('MINIAPP_ORIGIN');
$key = getenv('MINIAPP_PROXY_KEY');
$upstream = getenv('MINIAPP_UPSTREAM');
if (!$origin || !$key || strlen($key) < 32 || !$upstream) {
    fail(503, 'temporarily_unavailable');
}
$clientOrigin = $_SERVER['HTTP_ORIGIN'] ?? '';
$site = $_SERVER['HTTP_SEC_FETCH_SITE'] ?? '';
$method = $_SERVER['REQUEST_METHOD'] ?? '';
if (($clientOrigin !== '' && $clientOrigin !== $origin)
    || ($site !== '' && $site !== 'same-origin')
    || ($method !== 'GET' && $clientOrigin !== $origin)) {
    fail(403, 'forbidden');
}
$route = $_GET['route'] ?? '';
$paths = ['session' => '/api/miniapp/session', 'settings' => '/api/groups/current/settings',
    'host-query' => '/api/host/query', 'logout' => '/api/miniapp/logout', 'host-role' => '/api/host/role',
    'host-group-link' => '/api/host/group-link', 'host-case' => '/api/host/case',
    'host-case-reverse' => '/api/host/case/reverse', 'host-case-review' => '/api/host/case/review',
    'host-rule' => '/api/host/rule', 'host-rule-test' => '/api/host/rule/test', 'host-model' => '/api/host/model', 'host-model-rebuild' => '/api/host/model/rebuild', 'host-operations' => '/api/host/operations'];
if (!is_string($route) || !isset($paths[$route])) fail(404, 'invalid_request');
if ((in_array($route, ['session', 'host-query', 'logout', 'host-group-link', 'host-case', 'host-rule-test', 'host-model'], true) && $method !== 'POST')
    || (in_array($route, ['host-case-reverse', 'host-case-review'], true) && $method !== 'PATCH')
    || ($route === 'settings' && !in_array($method, ['GET', 'PATCH'], true))
    || (in_array($route, ['host-role', 'host-rule', 'host-model-rebuild', 'host-operations'], true) && !in_array($method, ['POST', 'PATCH'], true))) {
    fail(405, 'invalid_request');
}
$headers = ['Origin: ' . $origin, 'X-SPB-Proxy-Key: ' . $key, 'Content-Type: application/json'];
if ($route !== 'session') {
    $authorization = $_SERVER['HTTP_AUTHORIZATION'] ?? '';
    if (!preg_match('/\ABearer [a-f0-9]{32}\z/D', $authorization)) fail(401, 'session_expired');
    $headers[] = 'Authorization: ' . $authorization;
}
$body = '';
if ($method !== 'GET') {
    if (strtolower(trim(explode(';', $_SERVER['CONTENT_TYPE'] ?? '')[0])) !== 'application/json') fail(415, 'invalid_request');
    if ((int) ($_SERVER['CONTENT_LENGTH'] ?? 0) > 20480) fail(413, 'invalid_request');
    $stream = fopen('php://input', 'rb');
    $body = stream_get_contents($stream, 20481);
    fclose($stream);
    if ($body === false || strlen($body) > 20480) fail(413, 'invalid_request');
}
if (!function_exists('curl_init')) fail(503, 'temporarily_unavailable');
$curl = curl_init(rtrim($upstream, '/') . $paths[$route]);
$response = '';
curl_setopt_array($curl, [
    CURLOPT_CUSTOMREQUEST => $method,
    CURLOPT_HTTPHEADER => $headers,
    CURLOPT_FOLLOWLOCATION => false,
    CURLOPT_CONNECTTIMEOUT => 3,
    CURLOPT_TIMEOUT => 25,
    CURLOPT_WRITEFUNCTION => static function ($curl, string $chunk) use (&$response): int {
        if (strlen($response) + strlen($chunk) > 1048576) return 0;
        $response .= $chunk;
        return strlen($chunk);
    },
]);
if (defined('CURLOPT_PROTOCOLS_STR')) {
    curl_setopt($curl, CURLOPT_PROTOCOLS_STR, 'http,https');
} else {
    curl_setopt($curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
}
if ($method !== 'GET') curl_setopt($curl, CURLOPT_POSTFIELDS, $body);
$ok = curl_exec($curl);
$status = (int) curl_getinfo($curl, CURLINFO_HTTP_CODE);
curl_close($curl);
if ($ok === false || $status < 200 || $status >= 500) fail(503, $method === 'PATCH' ? 'save_failed' : 'temporarily_unavailable');
if (!in_array($status, [200, 400, 401, 403, 404, 409, 413, 429], true)) fail(502, 'temporarily_unavailable');
if (!is_array(json_decode($response, true))) fail(502, 'temporarily_unavailable');
http_response_code($status);
echo $response;
