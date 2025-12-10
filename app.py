# app.py
import os
import time
import tempfile
import hashlib
from pathlib import Path
import json  # necessário para salvar/ler lista de labels como JSON

import pandas as pd
import streamlit as st

# ------------------------- Configuration --------------------------
DATA_FILE = "data/frases-musicas-final-minimal.csv"   # sentenças (fallback)
PARQUET_FILE = "data/frases-musicas-final-filled-selected.parquet"  # versão Parquet (preferida)
ASSIGNMENTS_FILE = "data/assignments_second_round.csv"            # colunas: annotator_id,frase_id
USERS_FILE = "data/users_hashed.csv"                 # colunas: username,password_hash

ANNOTATIONS_DIR = Path("data/annotations")           # um arquivo por usuário
ANNOTATION_PREFIX = "annotation_second_round_"       # annotation_USERNAME.csv

# FASE 1 DE ANOTAÇÃO
# QUESTION_TEXT = "A sentença a seguir caracteriza ou qualifica uma pessoa?"
# LABELS = {
#     "sim": "✅ Sim",
#     "nao": "❌ Não",
#     "nao_sei": "🤔 Não sei/Indeterminado",
# }

# FASE 2 DE ANOTAÇÃO
QUESTION_TEXT = "A qual das categorias a sentença pertence?"
LABELS = {
    "identidade": "Identidade",
    "aparencia": "Aparência",
    "papel_social": "Papel social",
    "habilidades": "Habilidades/Competências",
    "valores": "Valores/Crenças",
    "comportamentos": "Comportamentos",
    "outros": "Outros",
}

# Autosave: salva quando atingir N mudanças OU após S segundos desde o último autosave
AUTOSAVE_EVERY = 10    # a cada N mudanças
AUTOSAVE_SECONDS = 15  # ou após N segundos

# Salvar explicitamente ao navegar? (Anterior/Próxima/Ir)
SAVE_ON_NAV = True   # False = não salva em toda navegação; respeita apenas o throttle

# ------------------------ Helper Functions ------------------------
def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def _select_required(df: pd.DataFrame, required_names: list[str], dtype_map: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Seleciona colunas exigidas independentemente de ordem e variações de caixa/espaços.
    Retorna já com os nomes finais (exatos em required_names) e dtypes aplicados se fornecidos.
    """
    norm = {str(c).strip().lower(): c for c in df.columns}
    missing = [col for col in required_names if col not in norm]
    if missing:
        raise ValueError(
            f"Arquivo não contém as colunas requeridas {required_names}. "
            f"Colunas encontradas: {list(df.columns)}"
        )
    out = df[[norm[c] for c in required_names]].copy()
    out.columns = required_names
    if dtype_map:
        out = out.astype(dtype_map)
    return out

def _path_mtime(path: str | Path) -> float:
    p = Path(path)
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0

def _atomic_write_csv(df: pd.DataFrame, final_path: Path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=str(final_path.parent), suffix=".tmp", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)  # atomic no mesmo FS

@st.cache_data(show_spinner=False)
def load_users(path: str = USERS_FILE, _mtime: float | None = None):
    # cache só invalida quando mtime muda (o valor participa da chave do cache)
    _mtime = _mtime if _mtime is not None else _path_mtime(path)
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str).fillna("")
    df = _select_required(
        df,
        required_names=["username", "password_hash"],
        dtype_map={"username": "string", "password_hash": "string"},
    )
    return dict(zip(df["username"], df["password_hash"]))

def authenticate(user, password):
    users = load_users(USERS_FILE, _mtime=_path_mtime(USERS_FILE))
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return users.get(str(user)) == password_hash

@st.cache_data(show_spinner=False)
def load_sentences(path_csv: str = DATA_FILE,
                   path_parquet: str = PARQUET_FILE,
                   _mtime_csv: float | None = None,
                   _mtime_parq: float | None = None):
    """
    Lê preferencialmente Parquet (colunas 'frase_id','frase').
    Cache é sensível a mtime de ambos os caminhos.
    """
    _mtime_csv = _mtime_csv if _mtime_csv is not None else _path_mtime(path_csv)
    _mtime_parq = _mtime_parq if _mtime_parq is not None else _path_mtime(path_parquet)

    p_parq = Path(path_parquet)
    if p_parq.exists():
        df = pd.read_parquet(p_parq, columns=["frase_id", "frase"])
        return df.astype({"frase_id": "string", "frase": "string"})

    # fallback: CSV (engine 'c' é mais rápido; se tiver CSV com aspas estranhas, troque para 'python')
    df = pd.read_csv(
        path_csv,
        dtype=str,
        engine="c",
        usecols=lambda c: str(c).strip().lower() in {"frase_id", "frase"},
    )
    norm = {str(c).strip().lower(): c for c in df.columns}
    df = df[[norm["frase_id"], norm["frase"]]].astype({"frase_id": "string", "frase": "string"})
    return df

@st.cache_data(show_spinner=False)
def load_assignments(path: str = ASSIGNMENTS_FILE, _mtime: float | None = None):
    _mtime = _mtime if _mtime is not None else _path_mtime(path)
    df = pd.read_csv(path, dtype=str, engine="c", usecols=["annotator_id", "frase_id"]).fillna("")
    df = _select_required(
        df,
        required_names=["annotator_id", "frase_id"],
        dtype_map={"annotator_id": "string", "frase_id": "string"},
    )
    return df

def get_user_annotations_file(username: str) -> Path:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_user = "".join(ch for ch in str(username) if ch.isalnum() or ch in ("_", "-", "."))
    return ANNOTATIONS_DIR / f"{ANNOTATION_PREFIX}{safe_user}.csv"

def load_user_annotations(username: str) -> pd.DataFrame:
    """
    Lê anotações do usuário. Compatível com arquivos antigos (sem 'sentiment').
    Retorna colunas: user, frase_id, annotation, sentiment
    """
    fpath = get_user_annotations_file(username)
    if not fpath.exists():
        return pd.DataFrame({
            "user": pd.Series(dtype="string"),
            "frase_id": pd.Series(dtype="string"),
            "annotation": pd.Series(dtype="string"),
            "sentiment": pd.Series(dtype="string"),
        })

    df = pd.read_csv(fpath, dtype=str, engine="c").fillna("")
    # compatibilidade: se não houver 'sentiment', adiciona como 'neutra'
    if "sentiment" not in df.columns:
        df["sentiment"] = "neutra"

    # garante colunas e tipos
    df = _select_required(
        df,
        required_names=["user", "frase_id", "annotation", "sentiment"],
        dtype_map={"user": "string", "frase_id": "string", "annotation": "string", "sentiment": "string"},
    )
    return df[df["user"] == str(username)]

def save_user_annotations(username: str, annotations_dict: dict, sentiment_dict: dict):
    """
    Sobrescreve de forma atômica (tmp + replace) o arquivo do usuário,
    incluindo a polaridade (sentiment).
    """
    fpath = get_user_annotations_file(username)

    if not annotations_dict and not sentiment_dict:
        df = pd.DataFrame(columns=["user", "frase_id", "annotation", "sentiment"])
        _atomic_write_csv(df, fpath)
        return

    rows = []

    # linhas onde há anotação (multilabel) -> inclui sentimento (ou neutra por padrão)
    for fid, ann in annotations_dict.items():
        sent = sentiment_dict.get(fid, "neutra")
        rows.append((username, fid, ann, sent))

    # linhas com sentimento mas sem anotação -> salva annotation como lista vazia "[]"
    for fid, sent in sentiment_dict.items():
        if fid not in annotations_dict:
            rows.append((username, fid, "[]", sent))

    df = pd.DataFrame(rows, columns=["user", "frase_id", "annotation", "sentiment"]) \
            .drop_duplicates(subset=["user", "frase_id"], keep="last")

    _atomic_write_csv(df, fpath)

# --------------------------- Main App -----------------------------
def main():
    st.title("Anotação de Sentenças")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # LOGIN
    if not st.session_state.authenticated:
        with st.form("login"):
            user = st.text_input("Usuário (ex.: user01, user02, ...)")
            password = st.text_input("Senha", type="password")
            submit = st.form_submit_button("Entrar")

            if submit:
                if authenticate(user, password):
                    st.session_state.authenticated = True
                    st.session_state.user = str(user).strip()
                    st.session_state.current_idx = 0  # índice dentro do conjunto atribuído
                    st.session_state.annotations_dict = {}  # {frase_id: JSON-string da lista de labels}
                    st.session_state.sentiment_dict = {}    # {frase_id: "positiva"/"negativa"/"neutra"}
                    st.session_state.unsaved_changes = 0
                    st.session_state.last_autosave_ts = 0.0
                    st.success(f"Bem-vindo, {st.session_state.user}!")
                    rerun()
                else:
                    st.error("Usuário ou senha incorretos.")
        return

    # SIDEBAR: Logout / Salvar agora
    with st.sidebar:
        if st.button("Logout"):
            if st.session_state.get("unsaved_changes", 0) > 0:
                save_user_annotations(
                    st.session_state.user,
                    st.session_state.annotations_dict,
                    st.session_state.sentiment_dict
                )
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            rerun()

        if st.button("💾 Salvar agora"):
            if st.session_state.get("unsaved_changes", 0) > 0:
                save_user_annotations(
                    st.session_state.user,
                    st.session_state.annotations_dict,
                    st.session_state.sentiment_dict
                )
                st.session_state.unsaved_changes = 0
                st.session_state.last_autosave_ts = time.time()
                st.success("Anotações salvas.")
            else:
                st.info("Nenhuma mudança pendente para salvar.")

    # CARREGAR DADOS BASE (com mtimes para estabilizar cache)
    try:
        sentences_df = load_sentences(
            path_csv=DATA_FILE,
            path_parquet=PARQUET_FILE,
            _mtime_csv=_path_mtime(DATA_FILE),
            _mtime_parq=_path_mtime(PARQUET_FILE),
        )
    except Exception as e:
        st.error(f"Erro ao ler sentenças: {e}")
        return

    try:
        assignments_df = load_assignments(
            path=ASSIGNMENTS_FILE,
            _mtime=_path_mtime(ASSIGNMENTS_FILE)
        )
    except Exception as e:
        st.error(f"Erro ao ler assignments: {e}")
        return

    # ---- Determinar as sentenças atribuídas ao usuário logado ----
    user_id_str = st.session_state.user
    user_assign = assignments_df[assignments_df["annotator_id"] == user_id_str]

    # fallback (caso assignments use número nu, tipo "1", "2"...)
    if user_assign.empty:
        try:
            user_id_int = int("".join(ch for ch in user_id_str if ch.isdigit()))
            user_assign = assignments_df[assignments_df["annotator_id"].astype(str) == str(user_id_int)]
        except Exception:
            pass

    if user_assign.empty:
        st.warning("Nenhuma sentença atribuída para este usuário em data/assignments.csv.")
        return

    # --- Materializa dados por sessão para evitar recomputações pesadas ---
    if "assigned_ids" not in st.session_state:
        st.session_state.assigned_ids = user_assign["frase_id"].astype(str).unique().tolist()
        st.session_state.assigned_ids_set = set(st.session_state.assigned_ids)

    if "frase_map" not in st.session_state:
        sent_small = sentences_df[sentences_df["frase_id"].isin(st.session_state.assigned_ids_set)]
        st.session_state.frase_map = dict(zip(sent_small["frase_id"], sent_small["frase"]))

    if "assigned_ids_filtered" not in st.session_state:
        st.session_state.assigned_ids_filtered = [
            fid for fid in st.session_state.assigned_ids
            if fid in st.session_state.frase_map
        ]
        if not st.session_state.assigned_ids_filtered:
            st.warning("Nenhuma sentença atribuída encontrada no arquivo de sentenças.")
            return

    # Carrega anotações pré-existentes do usuário (apenas 1x se possível)
    if "loaded_user_annotations" not in st.session_state:
        user_ann_df = load_user_annotations(user_id_str)
        stored_annotations = dict(zip(user_ann_df["frase_id"], user_ann_df["annotation"]))
        stored_sentiments = dict(zip(user_ann_df["frase_id"], user_ann_df["sentiment"]))
        st.session_state.annotations_dict = {**stored_annotations, **st.session_state.get("annotations_dict", {})}
        st.session_state.sentiment_dict = {**stored_sentiments, **st.session_state.get("sentiment_dict", {})}
        st.session_state.loaded_user_annotations = True

    # ---------- índice/registro corrente ----------
    assigned = st.session_state.assigned_ids_filtered
    total_assigned = len(assigned)

    idx = st.session_state.get("current_idx", 0)
    if idx >= total_assigned:
        idx = total_assigned - 1
        st.session_state.current_idx = idx

    current_frase_id = assigned[idx]
    current_sentence = st.session_state.frase_map.get(current_frase_id, "")

    # ---------- EXIBIÇÃO DA SENTENÇA E PERGUNTAS ----------
    # Cabeçalho com número e frase_id (acima das perguntas)
    st.write(f"##### Sentença {idx + 1} de {total_assigned} (frase_id: {current_frase_id})")

    # Sentença 
    st.markdown(
        f"""
        <div style="padding: 1rem; margin: 0.5rem 0; border: 2px solid #4a90e2; border-radius: 8px; background-color: #f0f6ff;">
            <p style="font-size:16px; font-weight:500; color:#222;">{current_sentence}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Pergunta principal 
    st.markdown("### A qual das categorias a sentença pertence?")

    # --- Linha de explicação das categorias---
    st.markdown(
        """
        <div style="
            margin: 0.25rem 0 1rem 0;
            padding: 0.8rem 1rem;
            background: #fffbe6;
            border: 1px solid #ffe58f;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.55;
            color: #333;
        ">
            <p><strong>Existem prescrições sobre o que define as pessoas em diferentes dimensões:</strong></p>
            <ul style="margin-top: 0.3rem; margin-bottom: 0; padding-left: 1.2rem;">
                <li><em>Identidade</em>: o que é ser, como a pessoa se define e é definida.</li>
                <li><em>Aparência</em>: quais atributos físicos são desejados ou indesejados.</li>
                <li><em>Papel social</em>: quais funções e posições sociais são atribuídas.</li>
                <li><em>Habilidades/Competências</em>: o que podem e devem fazer, capacidades esperadas.</li>
                <li><em>Valores/Crenças</em>: quais princípios, normas e ideais devem reger suas vidas.</li>
                <li><em>Comportamentos</em>: como devem agir, reagir ou se expressar na sociedade.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- CHECKBOXES (Fase 2) ----------
    annotations_dict = st.session_state.get("annotations_dict", {})
    current_value_raw = annotations_dict.get(current_frase_id, "")
    options = list(LABELS.keys())

    def _parse_labels(val):
        if isinstance(val, (list, tuple)):
            cand = list(val)
        elif isinstance(val, str) and val:
            try:
                parsed = json.loads(val)
                cand = parsed if isinstance(parsed, list) else []
            except Exception:
                cand = []
        else:
            cand = []
        return [x for x in cand if x in options]

    current_labels = _parse_labels(current_value_raw)

    cols = st.columns(3)
    new_selection = []
    for i, key in enumerate(options):
        ck = f"cb_{current_frase_id}_{key}"
        with cols[i % 3]:
            checked = st.checkbox(LABELS[key], value=(key in current_labels), key=ck)
            if checked:
                new_selection.append(key)

    # Atualiza sessão + autosave (multilabel) — permite ficar vazio, pois a validação será na navegação
    if new_selection != current_labels:
        if new_selection:
            st.session_state.annotations_dict[current_frase_id] = json.dumps(new_selection, ensure_ascii=False)
        else:
            # permite "limpar" a seleção atual
            st.session_state.annotations_dict[current_frase_id] = json.dumps([], ensure_ascii=False)

        st.session_state.unsaved_changes = st.session_state.get("unsaved_changes", 0) + 1
        now = time.time()
        should_time_save = (now - st.session_state.get("last_autosave_ts", 0)) >= AUTOSAVE_SECONDS
        should_count_save = st.session_state.unsaved_changes >= AUTOSAVE_EVERY
        if should_time_save or should_count_save:
            save_user_annotations(
                st.session_state.user,
                st.session_state.annotations_dict,
                st.session_state.sentiment_dict
            )
            st.session_state.unsaved_changes = 0
            st.session_state.last_autosave_ts = now
            st.toast("Autosave concluído.", icon="💾")

    # ---------- RADIO: polaridade da sentença (mesmo tamanho de título) ----------
    st.markdown("### A sentença é Positiva, Negativa ou Neutra?")
    sentiment_options = ["positiva", "negativa", "neutra"]
    current_sent = st.session_state.get("sentiment_dict", {}).get(current_frase_id, "neutra")
    chosen_sent = st.radio(
        label="Sentimento da sentença",
        options=sentiment_options,
        index=sentiment_options.index(current_sent) if current_sent in sentiment_options else 2,  # padrão: neutra
        horizontal=True,
        key=f"sent_{current_frase_id}",
        label_visibility="collapsed"
    )

    # Atualiza sentimento + autosave (polaridade)
    if st.session_state.get("sentiment_dict", {}).get(current_frase_id, "neutra") != chosen_sent:
        st.session_state.setdefault("sentiment_dict", {})[current_frase_id] = chosen_sent
        st.session_state.unsaved_changes = st.session_state.get("unsaved_changes", 0) + 1
        now = time.time()
        should_time_save = (now - st.session_state.get("last_autosave_ts", 0)) >= AUTOSAVE_SECONDS
        should_count_save = st.session_state.unsaved_changes >= AUTOSAVE_EVERY
        if should_time_save or should_count_save:
            save_user_annotations(
                st.session_state.user,
                st.session_state.annotations_dict,
                st.session_state.sentiment_dict
            )
            st.session_state.unsaved_changes = 0
            st.session_state.last_autosave_ts = now
            st.toast("Autosave concluído.", icon="💾")

    # ----------------- Progresso -----------------
    annotated_in_assigned = sum(1 for fid in assigned if fid in st.session_state.annotations_dict)
    st.progress(annotated_in_assigned / max(1, total_assigned))
    st.write(f"Você já anotou {annotated_in_assigned} de {total_assigned} sentenças atribuídas a você.")
    if st.session_state.get("unsaved_changes", 0) > 0:
        st.caption(f"Alterações pendentes: {st.session_state.unsaved_changes} (salvas automaticamente por tempo/contagem, ao clicar em 'Salvar agora' ou no logout).")

    # ---------- NAVEGAÇÃO ----------
    def maybe_save_on_nav():
        """Salva conforme política de navegação e throttle."""
        now = time.time()
        should_time_save = (now - st.session_state.get("last_autosave_ts", 0)) >= AUTOSAVE_SECONDS
        should_count_save = st.session_state.get("unsaved_changes", 0) >= AUTOSAVE_EVERY
        if SAVE_ON_NAV or should_time_save or should_count_save:
            if st.session_state.get("unsaved_changes", 0) > 0:
                save_user_annotations(
                    st.session_state.user,
                    st.session_state.annotations_dict,
                    st.session_state.sentiment_dict
                )
                st.session_state.unsaved_changes = 0
                st.session_state.last_autosave_ts = now

    def _has_valid_selection(fid: str) -> bool:
        """Retorna True se houver ao menos uma categoria marcada (persistida) para a frase."""
        val = st.session_state.get("annotations_dict", {}).get(fid, "")
        try:
            labs = json.loads(val) if isinstance(val, str) else val
        except Exception:
            labs = []
        return isinstance(labs, list) and len([x for x in labs if x in LABELS]) > 0

    col_prev, col_next = st.columns(2)
    if col_prev.button("⬅️ Anterior", key=f"prev_{idx}"):
        maybe_save_on_nav()
        if idx > 0:
            st.session_state.current_idx = idx - 1
        rerun()

    if col_next.button("Próxima ➡️", key=f"next_{idx}"):
        # Verifica apenas no clique para avançar
        if not _has_valid_selection(current_frase_id):
            st.warning("⚠️ Selecione ao menos uma categoria antes de avançar.")
        else:
            maybe_save_on_nav()
            if idx < total_assigned - 1:
                st.session_state.current_idx = idx + 1
            rerun()

    # Salto para próxima não anotada
    if st.button("⏭️ Pular para a próxima não anotada", key="jump_next_unann"):
        if not _has_valid_selection(current_frase_id):
            st.warning("⚠️ Selecione ao menos uma categoria antes de avançar.")
        else:
            maybe_save_on_nav()
            next_unann = None
            for i, fid in enumerate(assigned):
                if fid not in st.session_state.annotations_dict:
                    next_unann = i
                    break
            if next_unann is not None:
                st.session_state.current_idx = next_unann
            else:
                st.info("Todas as suas sentenças atribuídas já estão anotadas. 🎉")
            rerun()

    # Navegação direta por número (1..total_assigned)
    with st.form("jump_form"):
        jump_to = st.number_input(
            "Ir para sentença #",
            min_value=1,
            max_value=total_assigned,
            value=idx + 1,
            step=1,
            key=f"jump_num_{total_assigned}"
        )
        if st.form_submit_button("Ir"):
            if not _has_valid_selection(current_frase_id):
                st.warning("⚠️ Selecione ao menos uma categoria antes de avançar.")
            else:
                maybe_save_on_nav()
                st.session_state.current_idx = int(jump_to) - 1
                rerun()

if __name__ == "__main__":
    main()