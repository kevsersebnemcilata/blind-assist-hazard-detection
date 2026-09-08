"""
sistemin calisirken hesapladigi tehlike ve yaklasma verilerini grafiklere doker.

"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_object_count(log_path: str) -> None:
    """Plot per-frame object count (left axis) and cumulative ID switches (right axis) — Figure 7."""
    df = pd.read_csv(log_path)
    if df.empty:
        print(f"[plot_object_count] File is empty: {log_path}")
        return

    df.columns = [c.strip().lower() for c in df.columns]

    if "frame" not in df.columns or "object_count" not in df.columns:
        print(
            f"[plot_object_count] Expected columns 'frame' and 'object_count' in {log_path}.\n"
            f"  Found: {list(df.columns)}"
        )
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Detected Object Count", color="steelblue")
    ax1.plot(df["frame"], df["object_count"], color="steelblue",
             linewidth=1.8, label="Object Count")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    if "id_switch_count" in df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Cumulative ID Switch Count", color="crimson")
        ax2.plot(df["frame"], df["id_switch_count"], color="crimson",
                 linewidth=1.5, linestyle="--", label="ID Switches (cumulative)")
        ax2.tick_params(axis="y", labelcolor="crimson")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")
        print("[plot_object_count] Column 'id_switch_count' not found — skipping secondary axis.")

    plt.title("Detected Object Count Over Time (Figure 7)")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "figure7_object_count.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_object_count] Saved → {out_path}")
    plt.show()


def plot_danger_score_histogram(df: pd.DataFrame) -> None:
    """Tehlike skorunun frekans dağılımını histogram olarak çizer."""
    fig, ax = plt.subplots(figsize=(9, 5))

    n_bins = 30
    counts, edges, patches = ax.hist(
        df["danger_score"], bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.5
    )

    # Eşik çizgisi (0.45)
    threshold = 0.45
    ax.axvline(x=threshold, color="crimson", linestyle="--", linewidth=1.8, label=f"Alert Threshold ({threshold})")

    # Eşik üstü çubukları kırmızıya boya
    for patch, left in zip(patches, edges[:-1]):
        if left >= threshold:
            patch.set_facecolor("crimson")
            patch.set_alpha(0.75)

    # Yüzde bilgisi
    above = (df["danger_score"] >= threshold).mean() * 100
    ax.text(0.98, 0.95, f"Eşik üstü: %{above:.1f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    ax.set_title("Tehlike Skoru Dağılımı (Histogram)")
    ax.set_xlabel("Danger Score")
    ax.set_ylabel("Frame Sayısı")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_danger_histogram.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_danger_score_histogram] Saved → {out_path}")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """motion, depth, delta_d, approach ve danger_score arasındaki korelasyon ısı haritası."""
    cols = ["motion_score", "depth_score", "delta_d", "approach_score", "danger_score"]
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        print("[plot_correlation_heatmap] Yeterli sütun bulunamadı.")
        return

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")

    labels = [c.replace("_", "\n") for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Her hücreye değeri yaz
    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.values[i, j]
            color = "black" if abs(val) < 0.7 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("Metrik Korelasyon Isı Haritası")
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_correlation_heatmap] Saved → {out_path}")
    plt.show()


def plot_class_risk_distribution(log_path: str) -> None:
    """Sınıf bazlı risk skoru dağılımını kutu grafik ile gösterir (risk_summary.csv)."""
    if not os.path.isfile(log_path):
        print(f"[plot_class_risk_distribution] Dosya bulunamadı: {log_path}")
        return

    df = pd.read_csv(log_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "class_name" not in df.columns or "risk_score" not in df.columns:
        print(f"[plot_class_risk_distribution] Beklenen sütunlar yok. Bulunanlar: {list(df.columns)}")
        return

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    classes = df.groupby("class_name")["risk_score"].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    data_per_class = [df[df["class_name"] == cls]["risk_score"].dropna().values for cls in classes]
    bp = ax.boxplot(data_per_class, patch_artist=True, notch=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(y=0.45, color="crimson", linestyle="--", linewidth=1.5, label="Alert Threshold (0.45)")
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_title("Sınıf Bazlı Risk Skoru Dağılımı")
    ax.set_ylabel("Risk Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_class_risk_distribution.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_class_risk_distribution] Saved → {out_path}")
    plt.show()


def plot_occlusion_over_time(log_path: str) -> None:
    """Gizli (occluded) nesne sayısını zaman içinde gösterir (occlusion_log.csv)."""
    if not os.path.isfile(log_path):
        print(f"[plot_occlusion_over_time] Dosya bulunamadı: {log_path}")
        return

    df = pd.read_csv(log_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "frame" not in df.columns or "hidden_object_count" not in df.columns:
        print(f"[plot_occlusion_over_time] Beklenen sütunlar yok. Bulunanlar: {list(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(df["frame"], df["hidden_object_count"], alpha=0.35, color="darkorange")
    ax.plot(df["frame"], df["hidden_object_count"], color="darkorange", linewidth=1.8, label="Gizli Nesne Sayısı")

    ax.set_title("Zamanla Gizli (Occluded) Nesne Sayısı")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Gizli Nesne Sayısı")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_occlusion_over_time.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_occlusion_over_time] Saved → {out_path}")
    plt.show()


def plot_danger_level_pie(df: pd.DataFrame) -> None:
    """Kaç frame'in TEHLIKE / DIKKAT / GUVENLI seviyesinde geçtiğini pasta grafik ile gösterir."""
    tehlike = (df["danger_score"] >= 0.45).sum()
    dikkat   = ((df["danger_score"] >= 0.30) & (df["danger_score"] < 0.45)).sum()
    guvenli  = (df["danger_score"] < 0.30).sum()

    labels = ["TEHLIKE\n(≥0.45)", "DIKKAT\n(0.30–0.45)", "GUVENLI\n(<0.30)"]
    sizes  = [tehlike, dikkat, guvenli]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    explode = (0.06, 0.03, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda p: f"%{p:.1f}\n({int(round(p * sum(sizes) / 100))} frame)",
        startangle=140, pctdistance=0.72,
        wedgeprops=dict(linewidth=1.2, edgecolor="white")
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Tehlike Seviyesi Dağılımı (Frame Bazlı)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_danger_level_pie.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_danger_level_pie] Saved → {out_path}")
    plt.show()


def plot_class_detection_count(log_path: str) -> None:
    """Her nesne sınıfının kaç kez yüksek riskle tespit edildiğini yatay çubuk grafikle gösterir."""
    if not os.path.isfile(log_path):
        print(f"[plot_class_detection_count] Dosya bulunamadı: {log_path}")
        return

    df = pd.read_csv(log_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "class_name" not in df.columns:
        print(f"[plot_class_detection_count] 'class_name' sütunu yok. Bulunanlar: {list(df.columns)}")
        return

    counts = df["class_name"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.55)))
    bars = ax.barh(counts.index, counts.values,
                   color=plt.cm.tab10(np.linspace(0, 1, len(counts))), edgecolor="white")

    for bar, val in zip(bars, counts.values):
        ax.text(val + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    ax.set_title("Sınıf Bazlı Yüksek Risk Tespit Sayısı", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tespit Sayısı")
    ax.set_ylabel("Nesne Sınıfı")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_class_detection_count.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_class_detection_count] Saved → {out_path}")
    plt.show()


def plot_cumulative_alerts(df: pd.DataFrame) -> None:
    """Zaman içinde kümülatif tehlike uyarı sayısını (danger_score >= 0.45) çizer."""
    threshold = 0.45
    df = df.copy()
    df["alert"] = (df["danger_score"] >= threshold).astype(int)
    df["cumulative_alerts"] = df["alert"].cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df["frame_num"], df["cumulative_alerts"], alpha=0.25, color="crimson")
    ax.plot(df["frame_num"], df["cumulative_alerts"], color="crimson", linewidth=2,
            label="Kümülatif Uyarı Sayısı")

    total = df["cumulative_alerts"].iloc[-1]
    ax.text(0.98, 0.05, f"Toplam uyarı: {total}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    ax.set_title("Kümülatif Tehlike Uyarı Sayısı (Danger Score ≥ 0.45)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame Numarası")
    ax.set_ylabel("Toplam Uyarı Sayısı")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_cumulative_alerts.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_cumulative_alerts] Saved → {out_path}")
    plt.show()


def plot_depth_vs_danger_scatter(df: pd.DataFrame) -> None:
    """Derinlik skoru ile tehlike skoru arasındaki ilişkiyi renklendirilmiş scatter plot ile gösterir."""
    depth_col = "depth_score" if "depth_score" in df.columns else None
    if depth_col is None:
        print("[plot_depth_vs_danger_scatter] 'depth_score' sütunu bulunamadı.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        df[depth_col], df["danger_score"],
        c=df["danger_score"], cmap="RdYlGn_r",
        alpha=0.55, s=15, linewidths=0
    )
    plt.colorbar(sc, ax=ax, label="Danger Score")

    ax.axhline(y=0.45, color="crimson", linestyle="--", linewidth=1.5, label="Tehlike Eşiği (0.45)")
    ax.set_title("Derinlik Skoru vs Tehlike Skoru", fontsize=13, fontweight="bold")
    ax.set_xlabel("Depth Score (Yakınlık)")
    ax.set_ylabel("Danger Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join("outputs", "figure_depth_vs_danger_scatter.png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot_depth_vs_danger_scatter] Saved → {out_path}")
    plt.show()


def main():
    """
    logs/ dizinindeki en son dosyasini bulur ve
    tehlike puanini zaman icinde grafik olarak gösterir.
    """
    log_files = glob.glob(os.path.join("logs", "hazard_log_*.csv"))
    if not log_files:
        print("No log files found in 'logs' folder.")
        return

    latest_log = max(log_files, key=os.path.getctime)
    print(f"Plotting latest log: {latest_log}")

    df = pd.read_csv(latest_log)

    if len(df) == 0:
        print("Log file is empty.")
        return

    os.makedirs("outputs", exist_ok=True)

    # --- Grafik 1: Ana tehlike metrikleri zaman serisi ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['frame_num'], df['danger_score'], label='Total Danger Score', color='red', linewidth=2)
    plt.plot(df['frame_num'], df['motion_score'], label='Motion Component (Near)', color='blue', alpha=0.5, linestyle='--')
    plt.plot(df['frame_num'], df['delta_d'], label='Temporal Depth Delta (ΔD)', color='orange', alpha=0.5, linestyle='--')
    plt.plot(df['frame_num'], df['approach_score'], label='Approach Component', color='purple', alpha=0.5, linestyle='--')
    plt.axhline(y=0.45, color='black', linestyle=':', label='Alert Threshold')
    plt.title("Hazard Detection Metrics Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Score (0.0 to 1.0)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    output_path = os.path.join("outputs", "recent_session_plot.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.show()

    # --- Grafik 2: Tehlike skoru histogramı ---
    plot_danger_score_histogram(df)

    # --- Grafik 3: Metrik korelasyon ısı haritası ---
    plot_correlation_heatmap(df)

    # --- Grafik 4: Sınıf bazlı risk dağılımı ---
    risk_csv = os.path.join("logs", "risk_summary.csv")
    plot_class_risk_distribution(risk_csv)

    # --- Grafik 5: Gizli nesne zamanla ---
    occlusion_csv = os.path.join("logs", "occlusion_log.csv")
    plot_occlusion_over_time(occlusion_csv)

    # --- Figure 7: Nesne sayısı + ID switch ---
    id_switch_csv = os.path.join("logs", "id_switches.csv")
    if os.path.isfile(id_switch_csv):
        plot_object_count(id_switch_csv)
    else:
        print(f"[main] id_switches.csv not found — skipping Figure 7 (run with ENABLE_BYTETRACK=True first).")

    # --- Grafik 8: Tehlike seviyesi pasta grafiği ---
    plot_danger_level_pie(df)

    # --- Grafik 9: Sınıf bazlı tespit sayısı ---
    risk_csv2 = os.path.join("logs", "risk_summary.csv")
    plot_class_detection_count(risk_csv2)

    # --- Grafik 10: Kümülatif tehlike uyarı sayısı ---
    plot_cumulative_alerts(df)

    # --- Grafik 11: Derinlik vs Tehlike skoru scatter ---
    plot_depth_vs_danger_scatter(df)


if __name__ == "__main__":
    main()
