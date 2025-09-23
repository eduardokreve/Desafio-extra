#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agente E.D.A. multiagente para o terminal, com integração opcional ao LangGraph.

Este script permite que um usuário carregue um arquivo CSV e faça perguntas em
linguagem natural sobre os dados. Ele implementa todas as operações básicas de
análise exploratória de dados (EDA), como descrição de colunas, histogramas,
frequências, correlações, detecção de outliers, tendências temporais e
conclusões automáticas. Além dos comandos tradicionais da CLI, há um comando
`chat` que, quando as dependências de LangChain e LangGraph estão disponíveis,
cria um agente supervisionador capaz de escolher a ferramenta correta para
responder qualquer pergunta do usuário usando um modelo de linguagem.

Para usar o modo `chat`, é necessário instalar `langchain` e `langgraph` e
configurar uma chave de API para o provedor de modelo de linguagem (por exemplo,
OpenAI). Caso essas dependências não estejam disponíveis, o comando `chat`
informará o usuário e encerrará.

O script mantém um arquivo de sessão `.eda_session.json` com o caminho do CSV
carregado e o identificador do dataset. A memória persistente de cada dataset
fica armazenada em `.eda_memory/<dataset_id>.json`, onde são salvos o shape,
perfil de colunas, outliers detectados, insights gerados e caminhos para
gráficos. Os gráficos são salvos em `plots/<dataset_id>/` como PNG.

Exemplo de uso:

    python eda_multiagent.py load dados.csv --id meu_dataset
    python eda_multiagent.py profile
    python eda_multiagent.py chat

"""

import json
import hashlib
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import typer
from rich.console import Console
from rich.table import Table
from rich import box

# Importa LangChain/LangGraph apenas se disponível. Caso contrário, os
# operadores relacionados ao modo chat não funcionarão.
try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
except Exception:
    tool = None
    ChatOpenAI = None
    create_react_agent = None

# === Configurações ===
app = typer.Typer(add_completion=False, help="Agente E.D.A. multiagente para o terminal.")
console = Console()

# Arquivo de sessão e pastas de memória/plots
SESSION_FILE = Path(".eda_session.json")
MEM_DIR = Path(".eda_memory")
PLOTS_DIR = Path("plots")

MEM_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Variáveis globais para o dataset carregado
df_global: Optional[pd.DataFrame] = None
dataset_id_global: Optional[str] = None
mem_global: Optional[dict] = None
graph_global = None

# === Funções auxiliares ===
def hash_path(p: str) -> str:
    """Gera um hash curto a partir do caminho de um arquivo."""
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:12]


def default_dataset_id(csv_path: Path) -> str:
    """Gera um id padrão com base no nome do arquivo e hash do caminho completo."""
    return csv_path.stem + "-" + hash_path(str(csv_path.resolve()))


def load_session() -> dict:
    """Carrega o JSON de sessão com informações do dataset ativo."""
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    return {}


def save_session(d: dict) -> None:
    """Salva o JSON de sessão."""
    SESSION_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def mem_path(dataset_id: str) -> Path:
    """Retorna o caminho do arquivo de memória para um dataset."""
    return MEM_DIR / f"{dataset_id}.json"


def load_memory(dataset_id: str) -> dict:
    """Carrega a memória persistente de um dataset ou cria uma estrutura vazia."""
    p = mem_path(dataset_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "dataset_id": dataset_id,
        "shape": {},
        "columns": {},
        "outliers": {},
        "insights": [],
        "plots": {},
        "updated_at": None,
    }


def save_memory(dataset_id: str, mem: dict) -> None:
    """Salva a memória persistente de um dataset."""
    mem_path(dataset_id).write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_ready() -> tuple[Path, str]:
    """
    Verifica se há dataset carregado na sessão e se o arquivo existe.
    Retorna caminho e id. Caso contrário, aborta a execução com erro.
    """
    sess = load_session()
    csv = sess.get("csv_path")
    dsid = sess.get("dataset_id")
    if not csv or not dsid:
        console.print(
            "[red]Nenhum dataset carregado. Use:[/red] [bold]load path/to.csv --id seu_id[/bold]"
        )
        raise typer.Exit(1)
    p = Path(csv)
    if not p.exists():
        console.print(f"[red]CSV não encontrado:[/red] {p}")
        raise typer.Exit(1)
    return p, dsid


def read_csv(csv_path: Path, sample_for_profile: Optional[int] = None) -> pd.DataFrame:
    """
    Lê um CSV usando DuckDB para arquivos grandes quando possível; caso contrário,
    usa pandas. Se sample_for_profile for informado, lê apenas essa amostra.
    """
    try:
        import duckdb  # type: ignore

        if sample_for_profile:
            q = (
                f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}') USING SAMPLE "
                f"{sample_for_profile} ROWS;"
            )
            return duckdb.query(q).to_df()
        else:
            q = f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}');"
            return duckdb.query(q).to_df()
    except Exception:
        # Fallback para pandas
        if sample_for_profile:
            df_iter = pd.read_csv(csv_path, chunksize=sample_for_profile, low_memory=False)
            return next(df_iter)
        return pd.read_csv(csv_path, low_memory=False)


def infer_kind(s: pd.Series) -> str:
    """Classifica o tipo de uma série como numérica ou categórica."""
    if pd.api.types.is_bool_dtype(s):
        return "categórico"
    if pd.api.types.is_numeric_dtype(s):
        return "numérico" if s.nunique(dropna=True) > 10 else "categórico"
    return "categórico" if s.dtype == "O" else str(s.dtype)


def ensure_plots_dir(dataset_id: str) -> Path:
    """Garante que o diretório de plots para o dataset exista e o retorna."""
    d = PLOTS_DIR / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_plot(dataset_id: str, filename: str) -> Path:
    """Salva o gráfico atual em PNG e fecha o plot.

    Usa tight_layout e bbox_inches="tight" para que o plot se ajuste bem ao
    arquivo gerado. Retorna o caminho do arquivo salvo.
    """
    out = ensure_plots_dir(dataset_id) / filename
    plt.tight_layout()
    plt.savefig(out.as_posix(), dpi=140, bbox_inches="tight")
    plt.close()
    return out


def iqr_bounds(s: pd.Series) -> tuple[float, float]:
    """Calcula limites inferior e superior usando IQR."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def print_table(rows: List[dict], title: str) -> None:
    """Imprime uma tabela bonita no terminal usando Rich."""
    if not rows:
        console.print("[yellow]Nada a exibir.[/yellow]")
        return
    cols = list(rows[0].keys())
    table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
    for c in cols:
        table.add_column(str(c))
    for r in rows:
        table.add_row(*[str(r.get(c, "")) for c in cols])
    console.print(table)


def ensure_dataset_loaded() -> None:
    """Carrega o dataset no cache global se ainda não estiver carregado."""
    global df_global, dataset_id_global, mem_global
    if df_global is None:
        csv_path, dsid = dataset_ready()
        df_global = read_csv(csv_path)
        dataset_id_global = dsid
        mem_global = load_memory(dsid)


# === Funções de operações EDA utilizadas pelos comandos e tools ===
def compute_profile(column: Optional[str] = None) -> List[dict]:
    """Computa o perfil de todas as colunas ou de uma coluna específica.

    Retorna uma lista de dicionários com estatísticas.
    """
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    rows: List[dict] = []
    cols_to_process = [column] if column else list(df.columns)
    for col in cols_to_process:
        s = df[col]
        kind = infer_kind(s)
        missing = int(s.isna().sum())
        row = {
            "coluna": col,
            "tipo": kind,
            "dtype": str(s.dtype),
            "n_únicos": int(s.nunique(dropna=True)),
            "faltantes": missing,
            "faltantes_%": round(100 * missing / len(df), 4),
        }
        if pd.api.types.is_numeric_dtype(s):
            # Usa numpy para performance
            values = s.values
            row.update(
                {
                    "mín": float(np.nanmin(values)),
                    "máx": float(np.nanmax(values)),
                    "média": float(np.nanmean(values)),
                    "mediana": float(np.nanmedian(values)),
                    "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "variância": float(np.nanvar(values, ddof=1)) if len(values) > 1 else float("nan"),
                }
            )
        vc = s.value_counts(dropna=True).head(1)
        if not vc.empty:
            row["moda"] = str(vc.index[0])
            row["freq_moda"] = int(vc.iloc[0])
        rows.append(row)
    return rows


def generate_histogram(column: str, bins: int = 60) -> str:
    """Gera um histograma para uma coluna numérica e retorna o caminho do PNG."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    if column not in df.columns:
        raise ValueError(f"Coluna não encontrada: {column}")
    s = df[column]
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError("A coluna não é numérica.")
    plt.figure()
    s.plot(kind="hist", bins=bins, alpha=0.8)
    plt.title(f"Distribuição de {column}")
    plt.xlabel(column)
    plt.ylabel("Contagem")
    out = save_plot(dataset_id_global, f"hist_{column}.png")  # type: ignore
    # Atualiza a memória com o caminho
    mem = mem_global  # type: ignore
    mem.setdefault("plots", {})[f"hist_{column}"] = out.as_posix()
    mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
    save_memory(dataset_id_global, mem)
    return f"Histograma salvo em {out.as_posix()}"


def generate_bar(column: str) -> str:
    """Gera gráfico de barras de frequências para coluna categórica."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    if column not in df.columns:
        raise ValueError(f"Coluna não encontrada: {column}")
    vc = df[column].astype(str).value_counts().head(30)
    plt.figure()
    vc.plot(kind="bar")
    plt.title(f"Frequências de {column} (top 30)")
    plt.xlabel(column)
    plt.ylabel("Contagem")
    out = save_plot(dataset_id_global, f"bar_{column}.png")  # type: ignore
    mem = mem_global  # type: ignore
    mem.setdefault("plots", {})[f"bar_{column}"] = out.as_posix()
    mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
    save_memory(dataset_id_global, mem)
    return f"Gráfico de barras salvo em {out.as_posix()}"


def compute_frequencies(column: str, top: Optional[int] = None, bottom: Optional[int] = None) -> List[dict]:
    """Computa frequências de valores de uma coluna, retornando top ou bottom N."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    if column not in df.columns:
        raise ValueError(f"Coluna não encontrada: {column}")
    vc = df[column].astype(str).value_counts()
    if top:
        vc = vc.head(top)
    elif bottom:
        vc = vc.tail(bottom)
    rows = [{"valor": idx, "contagem": int(cnt)} for idx, cnt in vc.items()]
    return rows


def compute_correlation(target: Optional[str] = None, top: int = 10) -> List[dict]:
    """Calcula correlações entre variáveis. Se target é fornecido, retorna top corr."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        raise ValueError("Não há colunas numéricas suficientes.")
    corr_m = num.corr()
    if target:
        if target not in corr_m.columns:
            raise ValueError(f"Target não encontrada nas colunas numéricas: {target}")
        series = corr_m[target].drop(labels=[c for c in [target] if c in corr_m.columns])
        series = series.reindex(series.abs().sort_values(ascending=False).index)
        rows = [
            {"variável": k, "correlação": float(v)} for k, v in series.head(top).items()
        ]
        # Salva barplot
        keys = [r["variável"] for r in rows]
        vals = [r["correlação"] for r in rows]
        plt.figure()
        plt.bar(keys, vals)
        plt.title(f"Top {top} | Correlação com {target}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Correlação")
        out = save_plot(dataset_id_global, f"top_corr_with_{target}.png")  # type: ignore
        mem = mem_global  # type: ignore
        mem.setdefault("plots", {})[f"top_corr_with_{target}"] = out.as_posix()
        mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
        save_memory(dataset_id_global, mem)
        return rows
    else:
        # Retorna top 20 pares com maior correlação absoluta
        tril = []
        cols = list(corr_m.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                tril.append((a, b, float(corr_m.loc[a, b])))
        tril.sort(key=lambda x: abs(x[2]), reverse=True)
        rows = [
            {"a": a, "b": b, "corr": c} for a, b, c in tril[: min(20, len(tril))]
        ]
        return rows


def generate_scatter(x: str, y: str, sample: int = 5000) -> str:
    """Gera um gráfico de dispersão entre duas colunas numéricas. Salva e retorna o caminho."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    for c in (x, y):
        if c not in df.columns:
            raise ValueError(f"Coluna não encontrada: {c}")
    d = df[[x, y]].dropna()
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    plt.figure()
    plt.scatter(d[x], d[y], s=6, alpha=0.6)
    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    out = save_plot(dataset_id_global, f"scatter_{x}_{y}.png")  # type: ignore
    mem = mem_global  # type: ignore
    mem.setdefault("plots", {})[f"scatter_{x}_{y}"] = out.as_posix()
    mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
    save_memory(dataset_id_global, mem)
    return f"Dispersão salva em {out.as_posix()}"


def generate_box_by(column: str, by: str) -> str:
    """Gera boxplot de uma coluna numérica por outra (categórica)."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    if column not in df.columns or by not in df.columns:
        raise ValueError("Coluna ou 'by' não encontrada.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError("A coluna não é numérica.")
    plt.figure()
    df.boxplot(column=column, by=by)
    plt.suptitle("")
    plt.title(f"{column} por {by}")
    plt.xlabel(by)
    plt.ylabel(column)
    out = save_plot(dataset_id_global, f"box_{column}_by_{by}.png")  # type: ignore
    mem = mem_global  # type: ignore
    mem.setdefault("plots", {})[f"box_{column}_by_{by}"] = out.as_posix()
    mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
    save_memory(dataset_id_global, mem)
    return f"Boxplot salvo em {out.as_posix()}"


def compute_trend(time_col: str, freq: str = "hour", agg: str = "count", y: Optional[str] = None) -> List[dict]:
    """Computa tendência temporal e salva gráfico. Retorna lista de pontos."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    if time_col not in df.columns:
        raise ValueError(f"Coluna de tempo não encontrada: {time_col}")
    s = df[time_col]
    # Se numérico, trata como segundos desde t0 para agrupamento por hora
    if pd.api.types.is_numeric_dtype(s):
        if freq == "hour":
            key = (s // 3600).astype(int)
        elif freq == "day":
            key = (s // (3600 * 24)).astype(int)
        else:
            key = s
        if agg == "count":
            grouped = key.value_counts().sort_index()
        elif agg in {"sum", "mean"} and y:
            if y not in df.columns:
                raise ValueError(f"Coluna y não encontrada: {y}")
            grouped = df[y].groupby(key).agg(agg)
        else:
            grouped = key.value_counts().sort_index()
        # Salva gráfico
        plt.figure()
        plt.plot(grouped.index.values, grouped.values)
        plt.title(f"Tendência ({agg}) por {freq} - {time_col}")
        plt.xlabel(freq)
        plt.ylabel(agg)
        out = save_plot(dataset_id_global, f"trend_{time_col}_{freq}_{agg}.png")  # type: ignore
        mem = mem_global  # type: ignore
        mem.setdefault("plots", {})[f"trend_{time_col}_{freq}_{agg}"] = out.as_posix()
        mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
        save_memory(dataset_id_global, mem)
        return [{"chave": int(idx), "valor": float(val)} for idx, val in grouped.items()]
    # Tenta converter para datetime
    try:
        t = pd.to_datetime(s, errors="raise", dayfirst=True)
    except Exception:
        t = pd.to_datetime(s, errors="coerce")
    if t.isna().all():
        raise ValueError("Não foi possível interpretar a coluna de tempo.")
    df_t = df.copy()
    df_t["_t"] = t
    if freq == "hour":
        key = df_t["_t"].dt.floor("H")
    elif freq == "day":
        key = df_t["_t"].dt.date
    elif freq == "month":
        key = df_t["_t"].dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError("freq inválida. Use hour|day|month.")
    if agg == "count":
        grouped = df_t.groupby(key).size()
    elif agg in {"sum", "mean"} and y:
        if y not in df_t.columns:
            raise ValueError(f"Coluna y não encontrada: {y}")
        grouped = df_t.groupby(key)[y].agg(agg)
    else:
        grouped = df_t.groupby(key).size()
    plt.figure()
    plt.plot(grouped.index.values, grouped.values)
    plt.title(f"Tendência ({agg}) por {freq} - {time_col}")
    plt.xlabel(freq)
    plt.ylabel(agg)
    out = save_plot(dataset_id_global, f"trend_{time_col}_{freq}_{agg}.png")  # type: ignore
    mem = mem_global  # type: ignore
    mem.setdefault("plots", {})[f"trend_{time_col}_{freq}_{agg}"] = out.as_posix()
    mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
    save_memory(dataset_id_global, mem)
    return [
        {"chave": str(idx), "valor": float(val)} for idx, val in grouped.items()
    ]


def compute_outliers(column: str) -> List[dict]:
    """Detecta outliers de uma coluna usando IQR e atualiza memória."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    if column not in df.columns:
        raise ValueError(f"Coluna não encontrada: {column}")
    s = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError("A coluna não é numérica.")
    low, up = iqr_bounds(s)
    mask = (s < low) | (s > up)
    count = int(mask.sum())
    pct = round(100 * count / len(s), 4)
    # Atualiza memória
    mem = mem_global  # type: ignore
    mem.setdefault("outliers", {})[column] = {
        "count": count,
        "lower": float(low),
        "upper": float(up),
    }
    mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
    save_memory(dataset_id_global, mem)
    return [
        {
            "coluna": column,
            "outliers": count,
            "limite_inf": round(float(low), 4),
            "limite_sup": round(float(up), 4),
            "pct": pct,
        }
    ]


def generate_conclusions() -> List[str]:
    """Gera conclusões automáticas com base nas análises realizadas."""
    ensure_dataset_loaded()
    df = df_global  # type: ignore
    mem = mem_global  # type: ignore
    bullets: List[str] = []
    # 1) Desbalanceamento para colunas binárias
    bin_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]
    for c in bin_cols:
        vc = df[c].value_counts()
        total = int(vc.sum())
        minority = int(vc.min())
        pct = 100.0 * minority / total if total else 0.0
        if pct < 10:
            bullets.append(
                f"Coluna '{c}' é desbalanceada: classe minoritária ~{pct:.3f}% de {total} linhas."
            )
    # 2) Top correlações com coluna 'Class'
    if "Class" in df.columns and pd.api.types.is_numeric_dtype(df["Class"]):
        num = df.select_dtypes(include=[np.number])
        corr_m = num.corr()
        if "Class" in corr_m.columns:
            series = corr_m["Class"].drop(labels=["Class"])
            series = series.reindex(series.abs().sort_values(ascending=False).index)
            head = series.head(10)
            bullets.append(
                "Variáveis mais associadas à 'Class': "
                + "; ".join([f"{k} ({v:.3f})" for k, v in head.items()])
            )
    # 3) Outliers em memória
    outliers_info: dict = mem.get("outliers", {}) if mem else {}
    for col, info in outliers_info.items():
        bullets.append(
            f"Outliers em '{col}': {info['count']} pontos fora de ["
            f"{info['lower']:.2f}, {info['upper']:.2f}]. Considere winsorizar, "
            "transformar (ex.: log) ou investigar os extremos."
        )
    # 4) Tendências temporais básicas para coluna 'Time' numérica
    if "Time" in df.columns and pd.api.types.is_numeric_dtype(df["Time"]):
        hours = (df["Time"] // 3600).astype(int)
        vc = hours.value_counts().sort_index()
        if not vc.empty:
            bullets.append(
                f"Pico de volume na hora {int(vc.idxmax())} (desde t0) com {int(vc.max())} registros."
            )
    # Atualiza memória
    if bullets:
        mem_insights = mem.get("insights", []) if mem else []
        # Preserva ordem e evita duplicados
        new_insights = list(dict.fromkeys(mem_insights + bullets))
        mem["insights"] = new_insights
        mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
        save_memory(dataset_id_global, mem)  # type: ignore
    return bullets


# === Definições de tools para LangGraph ===
if tool is not None:
    @tool
    def describe_columns(column: str = "") -> str:
        """
        Retorna estatísticas descritivas do dataset ou de uma coluna específica.

        Se column for vazio, retorna descrição de todas as colunas; caso contrário,
        retorna estatísticas da coluna indicada. As estatísticas incluem mínimo,
        máximo, média, mediana, desvio padrão, variância, moda e contagem de
        faltantes.
        """
        try:
            rows = compute_profile(column if column else None)
        except Exception as e:
            return f"Erro ao descrever colunas: {e}"
        # Constrói texto amigável
        lines = []
        for r in rows:
            col = r["coluna"]
            tipo = r.get("tipo")
            linhas = [f"Coluna '{col}': tipo {tipo}"]
            if "mín" in r:
                linhas.append(
                    f"mín: {r['mín']:.4f}, máx: {r['máx']:.4f}, média: {r['média']:.4f}, "
                    f"mediana: {r['mediana']:.4f}, desvio-padrão: {r['std']:.4f}, variância: {r['variância']:.4f}"
                )
            linhas.append(
                f"n únicos: {r['n_únicos']}, faltantes: {r['faltantes']} "
                f"({r['faltantes_%']:.2f}%)"
            )
            if "moda" in r:
                linhas.append(f"moda: {r['moda']} (freq: {r['freq_moda']})")
            lines.append("; ".join(linhas))
        return "\n".join(lines)

    @tool
    def histogram(column: str, bins: int = 60) -> str:
        """
        Gera e salva um histograma de uma coluna numérica. Retorna o caminho do arquivo.
        """
        try:
            return generate_histogram(column, bins)
        except Exception as e:
            return f"Erro ao gerar histograma: {e}"

    @tool
    def bar_chart(column: str) -> str:
        """
        Gera e salva um gráfico de barras com as frequências de uma coluna.
        """
        try:
            return generate_bar(column)
        except Exception as e:
            return f"Erro ao gerar gráfico de barras: {e}"

    @tool
    def frequencies(column: str, top: int = 10, bottom: int = 0) -> str:
        """
        Devolve as frequências dos valores de uma coluna. Use 'top' ou 'bottom' para limitar.
        """
        try:
            rows = compute_frequencies(column, top if top > 0 else None, bottom if bottom > 0 else None)
            lines = [f"{r['valor']}: {r['contagem']}" for r in rows]
            return "\n".join(lines)
        except Exception as e:
            return f"Erro ao calcular frequências: {e}"

    @tool
    def correlation(target: str = "", top: int = 10) -> str:
        """
        Calcula correlações entre variáveis. Se um alvo for fornecido, mostra as top
        correlações dessa coluna; caso contrário, lista os pares mais correlacionados.
        """
        try:
            rows = compute_correlation(target if target else None, top)
            if target:
                lines = [f"{r['variável']}: {r['correlação']:.4f}" for r in rows]
            else:
                lines = [f"{r['a']} x {r['b']}: {r['corr']:.4f}" for r in rows]
            return "\n".join(lines)
        except Exception as e:
            return f"Erro ao calcular correlações: {e}"

    @tool
    def scatter_plot(x: str, y: str, sample: int = 5000) -> str:
        """
        Gera e salva um gráfico de dispersão entre duas colunas numéricas. O argumento
        'sample' limita a quantidade de pontos plotados.
        """
        try:
            return generate_scatter(x, y, sample)
        except Exception as e:
            return f"Erro ao gerar dispersão: {e}"

    @tool
    def boxplot(column: str, by: str) -> str:
        """
        Gera e salva um boxplot de uma coluna numérica agrupada por uma coluna categórica.
        """
        try:
            return generate_box_by(column, by)
        except Exception as e:
            return f"Erro ao gerar boxplot: {e}"

    @tool
    def trend(time_col: str, freq: str = "hour", agg: str = "count", y: str = "") -> str:
        """
        Gera tendência temporal a partir de uma coluna de tempo. 'freq' pode ser
        hour, day ou month. 'agg' define a métrica (count, sum ou mean). Quando
        agg é sum ou mean, forneça a coluna y.
        """
        try:
            points = compute_trend(time_col, freq, agg, y if y else None)
            return f"Trend calculado com {len(points)} pontos. Gráfico salvo em plots/{dataset_id_global}"
        except Exception as e:
            return f"Erro ao calcular tendência: {e}"

    @tool
    def outliers(column: str) -> str:
        """
        Detecta outliers de uma coluna numérica usando IQR. Retorna contagem e limites.
        """
        try:
            rows = compute_outliers(column)
            r = rows[0]
            return (
                f"Outliers em '{r['coluna']}': {r['outliers']} (\u2264 {r['limite_inf']}, \u2265 {r['limite_sup']})"
            )
        except Exception as e:
            return f"Erro ao detectar outliers: {e}"

    @tool
    def conclusions_tool() -> str:
        """
        Gera e retorna conclusões automáticas com base nas análises executadas até o momento.
        """
        bullets = generate_conclusions()
        if not bullets:
            return "Ainda não há conclusões; execute algumas análises antes."
        return "\n".join([f"• {b}" for b in bullets])


# === Função para construir o grafo (somente se dependências disponíveis) ===
def get_graph():
    global graph_global
    if graph_global is not None:
        return graph_global
    if ChatOpenAI is None or create_react_agent is None:
        return None

    ensure_dataset_loaded()

    tools_list = [
        describe_columns,
        histogram,
        bar_chart,
        frequencies,
        correlation,
        scatter_plot,
        boxplot,
        trend,
        outliers,
        conclusions_tool,
    ]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]Chave de API do OpenAI não encontrada. Ela é necessária para o modo chat.[/yellow]")
        api_key = console.input("Digite sua OpenAI API key: ").strip()
        os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    graph_global = create_react_agent(llm, tools_list)
    return graph_global


# === Comandos da CLI ===
@app.command()
def load(
    csv: str = typer.Argument(..., help="Caminho para o CSV"),
    id: Optional[str] = typer.Option(None, "--id", help="Identificador amigável do dataset"),
) -> None:
    """Carrega um CSV e inicializa memória persistente."""
    csv_path = Path(csv)
    if not csv_path.exists():
        console.print(f"[red]Arquivo não encontrado:[/red] {csv_path}")
        raise typer.Exit(1)
    dataset_id = id or default_dataset_id(csv_path)
    sess = {"csv_path": csv_path.as_posix(), "dataset_id": dataset_id}
    save_session(sess)
    # Carrega dataset completo no cache e atualiza shape em memória
    df = read_csv(csv_path, sample_for_profile=None)
    global df_global, dataset_id_global, mem_global, graph_global
    df_global = df
    dataset_id_global = dataset_id
    graph_global = None  # Reinicia grafo, pois dataset mudou
    mem = load_memory(dataset_id)
    mem["shape"] = {"rows": int(len(df)), "cols": int(df.shape[1])}
    mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
    save_memory(dataset_id, mem)
    mem_global = mem
    console.print(
        f"[green]OK![/green] Dataset id: [bold]{dataset_id}[/bold] | Shape: {len(df)} x {df.shape[1]}"
    )


@app.command()
def profile(
    column: Optional[str] = typer.Option(None, "--column", help="Coluna específica a descrever"),
) -> None:
    """Exibe o perfil das colunas ou de uma coluna específica."""
    try:
        rows = compute_profile(column)
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")
        raise typer.Exit(1)
    print_table(rows, title="Perfil das Colunas" if not column else f"Perfil da Coluna {column}")
    # Atualiza memória persistente com perfil das colunas
    if not column:
        mem = mem_global  # type: ignore
        mem["columns"] = {r["coluna"]: {k: r[k] for k in r if k != "coluna"} for r in rows}
        mem["updated_at"] = pd.Timestamp.utcnow().isoformat()
        save_memory(dataset_id_global, mem)  # type: ignore


@app.command()
def hist(
    column: str = typer.Argument(..., help="Coluna numérica para o histograma"),
    bins: int = typer.Option(60, "--bins", help="Número de bins do histograma"),
) -> None:
    """Gera histograma de uma coluna numérica e salva em plots."""
    try:
        msg = generate_histogram(column, bins)
        console.print(f"[green]{msg}[/green]")
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command("bar")
def bar_cmd(column: str = typer.Argument(..., help="Coluna categórica")) -> None:
    """Gera gráfico de barras para coluna categórica."""
    try:
        msg = generate_bar(column)
        console.print(f"[green]{msg}[/green]")
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command()
def freq(
    column: str = typer.Argument(..., help="Coluna para frequências"),
    top: Optional[int] = typer.Option(None, "--top", help="Mostrar top N"),
    bottom: Optional[int] = typer.Option(None, "--bottom", help="Mostrar bottom N"),
) -> None:
    """Exibe frequências de uma coluna, com top ou bottom N."""
    try:
        rows = compute_frequencies(column, top, bottom)
        print_table(rows, title=f"Frequências - {column}")
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command()
def corr(
    target: Optional[str] = typer.Option(None, "--target", help="Coluna alvo"),
    top: int = typer.Option(10, "--top", help="Número de correlações a mostrar"),
) -> None:
    """Calcula correlações. Se alvo é fornecido, mostra top correlacionadas com ele."""
    try:
        rows = compute_correlation(target, top)
        if target:
            title = f"Top {top} correlações com {target}"
            print_table(rows, title=title)
        else:
            title = "Top pares mais correlacionados"
            print_table(rows, title=title)
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command()
def scatter(
    x: str = typer.Argument(..., help="Coluna X"),
    y: str = typer.Argument(..., help="Coluna Y"),
    sample: int = typer.Option(5000, "--sample", help="Amostra para o gráfico de dispersão"),
) -> None:
    """Gera gráfico de dispersão entre duas colunas."""
    try:
        msg = generate_scatter(x, y, sample)
        console.print(f"[green]{msg}[/green]")
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command("box-by")
def box_by_cmd(
    column: str = typer.Argument(..., help="Coluna numérica"),
    by: str = typer.Option(..., "--by", help="Coluna categórica"),
) -> None:
    """Gera boxplot de uma coluna por outra."""
    try:
        msg = generate_box_by(column, by)
        console.print(f"[green]{msg}[/green]")
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command()
def trend_cmd(
    time_col: str = typer.Argument(..., help="Coluna de tempo"),
    freq: str = typer.Option("hour", "--freq", help="hour/day/month"),
    agg: str = typer.Option("count", "--agg", help="count|sum|mean"),
    y: Optional[str] = typer.Option(None, "--y", help="Coluna numérica para sum/mean"),
) -> None:
    """Calcula tendência temporal e gera gráfico."""
    try:
        points = compute_trend(time_col, freq, agg, y)
        console.print(
            f"[green]Trend calculado com {len(points)} pontos. Gráfico salvo em plots/{dataset_id_global}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command()
def outliers_cmd(column: str = typer.Argument(..., help="Coluna numérica")) -> None:
    """Detecta outliers de uma coluna e exibe resultados."""
    try:
        rows = compute_outliers(column)
        print_table(rows, title="Outliers (IQR)")
    except Exception as e:
        console.print(f"[red]Erro:[/red] {e}")


@app.command()
def conclusions() -> None:
    """Mostra e atualiza as conclusões automáticas."""
    bullets = generate_conclusions()
    if not bullets:
        console.print(
            "[yellow]Ainda não há insights suficientes; execute algumas análises antes.[/yellow]"
        )
    else:
        console.print("\n[bold]Conclusões do agente:[/bold]")
        for b in bullets:
            console.print(f"• {b}")


@app.command("memory")
def memory_show() -> None:
    """Exibe a memória persistida do dataset."""
    ensure_dataset_loaded()
    mem = mem_global  # type: ignore
    console.print_json(data=mem)


@app.command()
def chat() -> None:
    """
    Inicia um loop de conversa em linguagem natural usando LangGraph. O usuário
    pode fazer perguntas arbitrárias sobre o dataset e o agente escolherá a
    ferramenta adequada para respondê-las. Digite 'sair' ou 'exit' para encerrar.
    """
    ensure_dataset_loaded()
    g = get_graph()
    if g is None:
        console.print(
            "[red]Funcionalidade de chat indisponível. Instale langchain e langgraph e configure uma chave de API para usar este modo.[/red]"
        )
        raise typer.Exit(1)
    console.print(
        "[green]Modo chat iniciado. Pergunte algo sobre o dataset ou digite 'sair' para sair.[/green]"
    )
    try:
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[green]Chat encerrado.[/green]")
                break
            if not prompt:
                continue
            if prompt.lower() in {"sair", "exit", "quit"}:
                console.print("[green]Chat encerrado.[/green]")
                break
            # Prepara a mensagem para o agente; `create_react_agent` espera dicionário
            try:
                result = g.invoke({"messages": [{"role": "user", "content": prompt}]})
            except Exception as e:
                console.print(f"[red]Erro ao processar pergunta:[/red] {e}")
                continue
            # O retorno é um dicionário contendo as mensagens trocadas; exibimos a última
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if messages:
                    last = messages[-1]
                    # Mensagens do LangChain/LangGraph são objetos (AIMessage, etc.)
                    reply = getattr(last, "content", None)
                    if reply:
                        console.print(reply)
                        # segue para próximo input do usuário
                        continue
                    # Caso excepcional: se vier como dict
                    if isinstance(last, dict):
                        reply = last.get("content") or last.get("text")
                        if reply:
                            console.print(reply)
                            continue
                        # fallback
                        console.print(result)
    finally:
        pass


if __name__ == "__main__":
    app()