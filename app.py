"""X 계정 기반 선물 추천기 (Streamlit)

- X API v2 (Bearer Token)로 공개 게시물 수집
- Claude로 취향 분석 → 선물 추천
"""
import json
import re
import urllib.parse

import requests
import streamlit as st
from anthropic import Anthropic

X_API = "https://api.x.com/2"
MODEL = "claude-sonnet-5-5"

st.set_page_config(page_title="X 선물 추천기", page_icon="🎁", layout="centered")


# ---------- 공통 유틸 ----------
def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def parse_json(text: str):
    """LLM 응답에서 JSON 블록만 추출."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("{"), raw.find("[")) if i != -1], default=0)
    return json.loads(raw[start:])


def normalize_handle(raw: str) -> str:
    """'@abc', 'x.com/abc', 'https://twitter.com/abc?s=20' 등을 'abc'로."""
    s = raw.strip()
    m = re.search(r"(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})", s)
    if m:
        return m.group(1)
    return s.lstrip("@").strip()


def clean_token(raw: str) -> str:
    t = raw.strip().strip('"').strip("'")
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t


def x_get(url, headers, params, what):
    """5xx는 재시도 후 api.twitter.com으로도 시도. 실패 시 진단 정보를 에러에 담는다."""
    import time
    alt = url.replace("https://api.x.com/", "https://api.twitter.com/")
    r = None
    for target in (url, alt):
        for attempt in range(2):
            r = requests.get(target, headers=headers, params=params, timeout=20)
            if r.status_code < 500:
                break
            time.sleep(1.5 * (attempt + 1))
        if r.status_code < 500:
            break
    if r.status_code != 200:
        hint = {
            401: "토큰이 잘못됐어요. Bearer Token을 다시 복사해 보세요.",
            402: "크레딧이 부족해요. console.x.com → Billing에서 충전하세요.",
            403: "권한/크레딧 문제일 수 있어요. 콘솔에서 앱 상태와 크레딧을 확인하세요.",
            429: "요청 한도 초과예요. 잠시 후 다시 시도하세요.",
        }.get(r.status_code, "X 서버 쪽 오류일 수 있어요." if r.status_code >= 500 else "")
        tok = headers.get("Authorization", "").replace("Bearer ", "")
        diag = (f"[진단] 토큰 길이 {len(tok)}자, 시작 {tok[:4]!r} / "
                f"server={r.headers.get('server', '-')}, "
                f"content-type={r.headers.get('content-type', '-')}, "
                f"x-transaction-id={r.headers.get('x-transaction-id', '-')}, "
                f"cf-ray={r.headers.get('cf-ray', '-')}")
        body = r.text[:300] or "(응답 본문 없음)"
        raise RuntimeError(f"{what} 실패 ({r.status_code}) {hint}\n{body}\n{diag}")
    return r


# ---------- X 수집 ----------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_x(handle: str, max_posts: int, include_replies: bool, _token: str):
    """핸들의 프로필 + 최근 게시물. 같은 입력은 24시간 캐시(재과금 방지)."""
    headers = {"Authorization": f"Bearer {_token}"}

    r = x_get(
        f"{X_API}/users/by/username/{handle}",
        headers,
        {
            "user.fields": "description,location,public_metrics,protected",
            "expansions": "pinned_tweet_id",
            "tweet.fields": "text",
        },
        "사용자 조회",
    )
    body = r.json()
    if "data" not in body:
        raise RuntimeError(f"계정을 찾을 수 없어요: {body.get('errors', body)}")
    user = body["data"]
    if user.get("protected"):
        raise RuntimeError("비공개 계정이라 게시물을 읽을 수 없어요.")
    pinned = [t["text"] for t in body.get("includes", {}).get("tweets", [])]

    posts, token = [], None
    while len(posts) < max_posts:
        params = {
            "max_results": min(100, max(5, max_posts - len(posts))),
            "tweet.fields": "created_at,public_metrics,referenced_tweets",
        }
        if not include_replies:
            params["exclude"] = "replies"
        if token:
            params["pagination_token"] = token
        r = x_get(f"{X_API}/users/{user['id']}/tweets", headers, params, "게시물 조회")
        page = r.json()
        for t in page.get("data", []):
            kind = "RT" if any(x["type"] == "retweeted" for x in t.get("referenced_tweets", []) or []) else "글"
            posts.append({"date": t.get("created_at", "")[:10], "kind": kind, "text": t["text"]})
        token = page.get("meta", {}).get("next_token")
        if not token:
            break

    return {"user": user, "pinned": pinned, "posts": posts[:max_posts]}


# ---------- Claude ----------
def ask_claude(client: Anthropic, prompt: str, max_tokens: int = 4000) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def analyze_profile(client, data):
    u = data["user"]
    posts_txt = "\n".join(f"[{p['date']}][{p['kind']}] {p['text']}" for p in data["posts"])
    prompt = f"""다음은 X 계정 @{u['username']} ({u.get('name','')})의 공개 프로필과 최근 게시물이야.
선물 추천을 위해 이 사람의 취향을 분석해줘.

규칙:
- 게시물에 근거가 있는 것만 써. 추측은 confidence를 낮게.
- "갖고 싶다/사고 싶다/품절/위시/탐난다" 같은 표현은 wishlist에 우선 반영.
- 이미 샀거나 가진 물건은 owned에 넣어 (추천에서 제외할 용도).
- 건강·정치·종교 등 민감한 특성은 추론하지 마.
- RT는 본인 글보다 약한 신호로 취급.

[소개] {u.get('description','')}
[위치] {u.get('location','')}
[고정 트윗] {' / '.join(data['pinned'])}

[게시물]
{posts_txt}

JSON만 출력:
{{
  "summary": "한두 문장 요약",
  "interests": [{{"name": "", "strength": "상/중/하", "evidence": "근거가 된 게시물 짧은 발췌"}}],
  "likes": ["좋아하는 브랜드·작품·캐릭터·음식 등"],
  "wishlist": [{{"item": "", "evidence": ""}}],
  "owned": ["이미 가진 것"],
  "dislikes": ["싫어하는 것"],
  "style": "선호 스타일/미감 (근거 있을 때만)"
}}"""
    return parse_json(ask_claude(client, prompt))


def recommend(client, handle, profile, budget, relation, occasion, extra):
    prompt = f"""@{handle} 에게 줄 선물을 추천해줘. 한국에서 구매 가능한 것 위주로.

[취향 분석]
{json.dumps(profile, ensure_ascii=False, indent=1)}

[조건]
- 예산: {budget:,}원 이하 (가격대는 대략적으로)
- 관계: {relation}
- 상황: {occasion}
- 추가 요청: {extra or '없음'}

규칙:
- wishlist에 있는 것이 예산 안이면 최우선.
- owned에 있는 것과 dislikes는 제외.
- 뻔한 범용 선물(상품권, 텀블러 등)은 최대 1개.
- 관계에 비해 부담스럽거나 너무 사적인 선물은 피해.
- 8개 추천, 확신도 높은 순.

JSON만 출력:
{{"recommendations": [{{
  "name": "선물 이름",
  "price": "예상 가격대",
  "why": "왜 좋아할지 (취향과 연결)",
  "evidence": "근거 게시물 짧은 발췌",
  "search_keyword": "쇼핑몰 검색용 키워드",
  "confidence": "상/중/하"
}}]}}"""
    return parse_json(ask_claude(client, prompt))["recommendations"]


def shop_links(kw: str) -> str:
    q = urllib.parse.quote(kw)
    return (f"[네이버쇼핑](https://search.shopping.naver.com/search/all?query={q}) · "
            f"[쿠팡](https://www.coupang.com/np/search?q={q})")


# ---------- 접근 제한 (secrets에 APP_PASSWORD가 있을 때만) ----------
app_pw = secret("APP_PASSWORD")
if app_pw and st.session_state.get("authed") is not True:
    st.title("🎁 X 선물 추천기")
    pw = st.text_input("비밀번호", type="password")
    if pw:
        if pw == app_pw:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("비밀번호가 달라요.")
    st.stop()


# ---------- UI ----------
st.title("🎁 X 선물 추천기")

with st.expander("🔑 API 키", expanded=not (secret("X_BEARER_TOKEN") and secret("ANTHROPIC_API_KEY"))):
    x_token = st.text_input("X Bearer Token", value=secret("X_BEARER_TOKEN"), type="password")
    claude_key = st.text_input("Anthropic API Key", value=secret("ANTHROPIC_API_KEY"), type="password")
    st.caption("입력한 키는 저장되지 않아요. 매번 입력하기 싫으면 Streamlit Secrets에 넣어두세요.")

handle = normalize_handle(st.text_input("X 계정", placeholder="@handle 또는 프로필 링크"))
x_token = clean_token(x_token)
claude_key = claude_key.strip()
if handle and not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle):
    st.warning("계정 아이디는 영문·숫자·밑줄 15자 이내예요. 표시 이름(닉네임)이 아니라 @ 뒤의 아이디를 넣어주세요.")
c1, c2 = st.columns(2)
budget = c1.number_input("예산(원)", min_value=5000, value=50000, step=10000)
relation = c2.selectbox("관계", ["친구", "연인", "가족", "직장 동료", "지인/팔로워"])
occasion = st.selectbox("상황", ["생일", "기념일", "크리스마스", "감사 선물", "집들이", "그냥"])
extra = st.text_input("추가 요청 (선택)", placeholder="예: 실용적인 걸로, 굿즈 말고")

with st.expander("⚙️ 수집 옵션"):
    max_posts = st.slider("읽을 게시물 수", 50, 500, 200, step=50)
    include_replies = st.checkbox("답글 포함", value=False)
    if st.button("🔌 X 연결 테스트 (@XDevelopers 조회)"):
        try:
            r = x_get(f"{X_API}/users/by/username/XDevelopers",
                      {"Authorization": f"Bearer {x_token}"}, {}, "연결 테스트")
            st.success(f"연결 OK: {r.json().get('data', {}).get('name')} — 문제는 입력한 계정 쪽이에요.")
        except Exception as e:
            st.error(str(e))
    st.caption(f"예상 X API 비용: 약 ${max_posts * 0.005:.2f} (게시물 1개당 $0.005 기준, 같은 조건은 24시간 캐시)")

if st.button("분석하고 추천받기", type="primary", use_container_width=True):
    if not (x_token and claude_key and handle):
        st.warning("키 두 개와 계정을 모두 입력해 주세요.")
        st.stop()
    client = Anthropic(api_key=claude_key)
    try:
        with st.status("작업 중...", expanded=True) as status:
            st.write("📥 게시물 수집 중")
            data = fetch_x(handle, max_posts, include_replies, x_token)
            if not data["posts"]:
                raise RuntimeError("읽을 수 있는 게시물이 없어요.")
            st.write(f"✅ {len(data['posts'])}개 수집")
            st.write("🧠 취향 분석 중")
            profile = analyze_profile(client, data)
            st.write("🎁 선물 고르는 중")
            recs = recommend(client, handle, profile, budget, relation, occasion, extra)
            status.update(label="완료", state="complete", expanded=False)
        st.session_state.result = {"handle": handle, "profile": profile, "recs": recs, "n": len(data["posts"])}
    except Exception as e:
        st.error(f"오류: {e}")

res = st.session_state.get("result")
if res:
    p = res["profile"]
    st.subheader(f"@{res['handle']} 취향 요약")
    st.write(p.get("summary", ""))
    if p.get("interests"):
        st.markdown("**관심사** · " + ", ".join(f"{i['name']}({i.get('strength','')})" for i in p["interests"]))
    if p.get("wishlist"):
        st.markdown("**위시 신호** · " + ", ".join(w["item"] for w in p["wishlist"]))
    with st.expander("분석 상세 보기"):
        st.json(p)

    st.subheader("추천 선물")
    for i, r in enumerate(res["recs"], 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r['name']}** · {r.get('price','')} · 확신도 {r.get('confidence','')}")
            st.write(r.get("why", ""))
            if r.get("evidence"):
                st.caption(f"근거: “{r['evidence']}”")
            st.markdown(shop_links(r.get("search_keyword") or r["name"]))
