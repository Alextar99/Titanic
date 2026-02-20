# ==============================================================
# ANÁLISIS EXPLORATORIO: House Prices – Advanced Regression
# Kaggle: house-prices-advanced-regression-techniques
# URL   : https://www.kaggle.com/competitions/
#         house-prices-advanced-regression-techniques
# ==============================================================


# ==============================================================
# 0. INSTALACIÓN Y CARGA DE LIBRERÍAS
# ==============================================================

rm(list=ls())
dev.off()

pkgs <- c(
  "tidyverse", "skimr", "naniar", "corrplot",
  "plotly",    "scales", "gridExtra", "moments",
  "RColorBrewer", "ggridges", "viridis", "ggrepel"
)
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

library(tidyverse)
library(skimr)
library(naniar)
library(corrplot)
library(plotly)
library(scales)
library(gridExtra)
library(moments)
library(RColorBrewer)
library(ggridges)
library(viridis)
library(ggrepel)

# Tema gráfico personalizado
theme_hp <- theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle = element_text(color = "grey40", size = 10),
    axis.title    = element_text(size = 10),
    legend.title  = element_text(face = "bold", size = 9),
    strip.text    = element_text(face = "bold")
  )


# ==============================================================
# a) PREPARACIÓN DE LOS DATOS
# ==============================================================

# ---------------------------------------------------------------
# a.1) DESCRIPCIÓN DEL CONJUNTO ORIGINAL
# ---------------------------------------------------------------
cat(strrep("=", 65), "\n")
cat("NOMBRE   : House Prices – Advanced Regression Techniques\n")
cat("FUENTE   : Kaggle\n")
cat("ENLACE   : https://www.kaggle.com/competitions/\n")
cat("           house-prices-advanced-regression-techniques\n")
cat("CONTEXTO : Precios de venta de viviendas residenciales\n")
cat("           en Ames, Iowa (EE.UU.) — años 2006-2010\n")
cat(strrep("=", 65), "\n\n")

# Cargar el dataset
train_raw <- read.csv("C:\\Users\\alega\\OneDrive\\Documentos\\Análisis de Datos\\Trabajo ADAT\\House-prices\\train.csv", 
                      header = TRUE, stringsAsFactors = FALSE)

n_obs  <- nrow(train_raw)    # individuos
n_vars <- ncol(train_raw)    # variables
n_num  <- sum(sapply(train_raw, is.numeric))
n_chr  <- sum(sapply(train_raw, is.character))

cat(sprintf("Número de individuos (filas)     : %d\n",   n_obs))
cat(sprintf("Número de variables (columnas)   : %d\n",   n_vars))
cat(sprintf("  - Variables numéricas          : %d\n",   n_num))
cat(sprintf("  - Variables de texto (categ.)  : %d\n\n", n_chr))

cat("Variables del dataset:\n")
print(names(train_raw))

cat("\nPrimeras 5 filas (primeras 10 columnas):\n")
print(head(train_raw[, 1:10], 5))


# ---------------------------------------------------------------
# a.2) TRANSFORMACIÓN, LIMPIEZA E IMPUTACIÓN
# ---------------------------------------------------------------
train <- train_raw  # copia de trabajo

# Renombrar variables con nombres no sintácticos en R
train <- train %>%
  rename(FlrSF_1st  = X1stFlrSF,
         FlrSF_2nd  = X2ndFlrSF,
         Porch_3Ssn = X3SsnPorch)


# --- PASO 1: Diagnóstico de valores perdidos -------------------
miss_df <- data.frame(
  Variable    = names(train),
  N_Missing   = colSums(is.na(train)),
  Pct_Missing = round(100 * colMeans(is.na(train)), 2)
) %>%
  filter(N_Missing > 0) %>%
  arrange(desc(Pct_Missing))

cat("\n--- Diagnóstico de valores perdidos (antes de imputar) ---\n")
print(miss_df, row.names = FALSE)

# Visualización de valores faltantes
p_miss <- gg_miss_var(train, show_pct = TRUE) +
  labs(
    title    = "Porcentaje de valores faltantes por variable",
    subtitle = sprintf("House Prices Dataset — Kaggle (n = %d obs., %d vars.)",
                       n_obs, n_vars),
    y = "% Missing"
  ) +
  theme_hp
print(p_miss)


# --- PASO 2: Imputación semántica (según data dictionary) ------

# 2a. Categóricas: NA = "No tiene esa característica"
na_none_cat <- c(
  "Alley",        "MasVnrType",   "BsmtQual",    "BsmtCond",
  "BsmtExposure", "BsmtFinType1", "BsmtFinType2","FireplaceQu",
  "GarageType",   "GarageFinish", "GarageQual",  "GarageCond",
  "PoolQC",       "Fence",        "MiscFeature"
)
for (v in na_none_cat) {
  train[[v]][is.na(train[[v]])] <- "None"
}

# 2b. Numéricas: NA = 0 (no posee la característica)
na_zero_num <- c(
  "MasVnrArea", "BsmtFinSF1",   "BsmtFinSF2",  "BsmtUnfSF",
  "TotalBsmtSF","BsmtFullBath", "BsmtHalfBath",
  "GarageCars", "GarageArea",   "GarageYrBlt"
)
for (v in na_zero_num) {
  train[[v]][is.na(train[[v]])] <- 0
}

# 2c. LotFrontage (~18% NA): imputación por mediana de Neighborhood
train <- train %>%
  group_by(Neighborhood) %>%
  mutate(LotFrontage = if_else(
    is.na(LotFrontage),
    median(LotFrontage, na.rm = TRUE),
    as.double(LotFrontage)
  )) %>%
  ungroup()

# 2d. Electrical (1 NA): imputar con la moda
moda_elec <- names(which.max(table(train$Electrical)))
train$Electrical[is.na(train$Electrical)] <- moda_elec


# --- PASO 3: Feature Engineering --------------------------------
# Extraer YrSold como entero ANTES de cualquier conversión a factor
# (as.integer(as.character()) funciona sea cual sea el tipo actual)
yr_num <- as.integer(as.character(train$YrSold))

train <- train %>%
  mutate(
    # Edad de la vivienda en el momento de la venta
    HouseAge     = yr_num - YearBuilt,
    # Años desde la última remodelación
    RemodAge     = yr_num - YearRemodAdd,
    # Superficie total: sótano + planta baja + planta alta
    TotalSF      = TotalBsmtSF + FlrSF_1st + FlrSF_2nd,
    # Nº total de baños (completos + 0.5 × medios baños)
    TotalBaths   = FullBath + 0.5 * HalfBath +
      BsmtFullBath + 0.5 * BsmtHalfBath,
    # Indicadores binarios
    HasPool      = factor(ifelse(PoolArea    > 0, "Sí", "No")),
    HasGarage    = factor(ifelse(GarageArea  > 0, "Sí", "No")),
    HasFireplace = factor(ifelse(Fireplaces  > 0, "Sí", "No")),
    HasBasement  = factor(ifelse(TotalBsmtSF > 0, "Sí", "No")),
    # Transformación logarítmica de la variable respuesta
    logSalePrice = log(SalePrice)
  )


train <- train %>%
  mutate(
    # La "feature reina": superficie ponderada por calidad
    QualSF = as.integer(OverallQual) * TotalSF,
    
    # Variantes también potentes
    QualGrLiv   = as.integer(OverallQual) * GrLivArea,  # solo área habitable
    QualBsmt    = as.integer(OverallQual) * TotalBsmtSF, # calidad × sótano
    QualGarage  = as.integer(OverallQual) * GarageArea,  # calidad × garaje
    
    # Puntuación compuesta de estado general (calidad + condición)
    OverallScore = as.integer(OverallQual) * as.integer(OverallCond),
    
    # Superficie total de porches (feature agregada útil)
    TotalPorchSF = OpenPorchSF + EnclosedPorch + Porch_3Ssn + ScreenPorch
  )

# Verificar correlación de las nuevas features con SalePrice
new_feats <- c("QualSF", "QualGrLiv", "QualBsmt", "QualGarage",
               "OverallScore", "TotalPorchSF", "TotalSF", "SalePrice")

cor_new <- cor(
  train %>% select(all_of(new_feats)),
  use = "pairwise.complete.obs"
)["SalePrice", ] %>%
  sort(decreasing = TRUE)

print(round(cor_new, 3))




# --- PASO 4: Conversión al tipo de dato correcto ----------------

# MSSubClass: código numérico que representa tipo de vivienda → factor nominal
train$MSSubClass <- factor(train$MSSubClass)

# OverallQual / OverallCond: escala ordinal 1–10
train$OverallQual <- factor(train$OverallQual, levels = 1:10, ordered = TRUE)
train$OverallCond <- factor(train$OverallCond, levels = 1:10, ordered = TRUE)

# MoSold: mes de venta → factor con etiquetas
train$MoSold <- factor(train$MoSold, levels = 1:12, labels = month.abb)

# YrSold: año de venta → factor
train$YrSold <- factor(train$YrSold)

# Variables de calidad con escala ordinal estandarizada
quality_levels  <- c("None", "Po", "Fa", "TA", "Gd", "Ex")
ord_qual_vars   <- c(
  "ExterQual", "ExterCond", "BsmtQual",   "BsmtCond",
  "HeatingQC", "KitchenQual","FireplaceQu",
  "GarageQual","GarageCond"
)
for (v in ord_qual_vars) {
  train[[v]] <- factor(train[[v]], levels = quality_levels, ordered = TRUE)
}

# Resto de variables de texto → factor nominal
chr_vars <- names(train)[sapply(train, is.character)]
train[chr_vars] <- lapply(train[chr_vars], factor)


# --- PASO 5: Verificación final de valores perdidos -------------
n_na_restantes <- sum(is.na(train))
cat(sprintf("\nValores perdidos tras la imputación: %d\n", n_na_restantes))
if (n_na_restantes > 0) {
  cat("Variables con NAs restantes:",
      paste(names(which(colSums(is.na(train)) > 0)), collapse = ", "), "\n")
}


# ---------------------------------------------------------------
# a.3) RESUMEN FINAL DEL CONJUNTO
# ---------------------------------------------------------------
cat("\n", strrep("=", 65), "\n")
cat("RESUMEN FINAL (TRAS PREPROCESADO)\n")
cat(strrep("=", 65), "\n")
cat(sprintf("Individuos                  : %d\n",   nrow(train)))
cat(sprintf("Variables totales           : %d\n",   ncol(train)))
cat(sprintf("  - Numéricas               : %d\n",   sum(sapply(train, is.numeric))))
cat(sprintf("  - Factor / Ordinal        : %d\n",   sum(sapply(train, is.factor))))
cat(sprintf("  - Variables creadas       : %s\n",
            "HouseAge, RemodAge, TotalSF, TotalBaths,\n"))
cat(sprintf("                              %s\n",
            "HasPool, HasGarage, HasFireplace, HasBasement, logSalePrice"))
cat(sprintf("  - Valores perdidos        : %d\n\n", sum(is.na(train))))

# Resumen estadístico completo con skimr
print(skim(train))


# ---------------------------------------------------------------
# a.3) RESUMEN FINAL DEL CONJUNTO
# ---------------------------------------------------------------
# ... (código existente sin tocar)
# print(skim(train))   ← última línea de a.3)

# ==============================================================
# a.4) TRATAMIENTO DE OUTLIERS (DATOS ATÍPICOS)
# ==============================================================
# Ref. teórica : Barnett & Lewis (1994) — definición de outlier
#                Tukey (1977)           — método IQR clásico
#                Hubert & Vandervieren (2008, Comput. Stat. & Data
#                  Analysis, 52, 5186–5201) — boxplot ajustado
#                Tema 2: Preparación de datos (Caballé & Alonso)
#
# NOTA FUNDAMENTAL: un dato atípico NO es necesariamente un error.
# Su inclusión o exclusión es una DECISIÓN ESTADÍSTICA (Barnett & Lewis).
# Este bloque se ejecuta ANTES del EDA para que todo el análisis
# exploratorio opere sobre los datos ya saneados.
#
# Metodología aplicada:
#   PASO 6  — Diagnóstico univariante: Tukey 1.5·IQR y 3·IQR
#   PASO 7  — Mosaico de boxplots diagnósticos
#   PASO 8  — Corrección por asimetría: boxplot ajustado H&V (2008)
#   PASO 9  — Análisis contextual del dominio inmobiliario
#   PASO 10 — Decisión estadística y winsorización [P0.5%, P99.5%]
#   PASO 11 — Verificación visual pre/post
#   PASO 12 — Gráfico interactivo: mapa de outliers (plotly)
# ==============================================================

# Instalar y cargar robustbase (función mc() para el medcouple)
if (!requireNamespace("robustbase", quietly = TRUE)) install.packages("robustbase")
library(robustbase)

# Variables numéricas clave a diagnosticar
out_vars <- c(
  "SalePrice",    "GrLivArea",   "LotArea",      "LotFrontage",
  "TotalSF",      "TotalBsmtSF", "FlrSF_1st",    "FlrSF_2nd",
  "GarageArea",   "TotalBaths",  "HouseAge",     "RemodAge",
  "Fireplaces",   "TotRmsAbvGrd","WoodDeckSF",   "OpenPorchSF",
  "TotalPorchSF", "QualSF"
)

# ---------------------------------------------------------------
# PASO 6 — Diagnóstico univariante: Tukey 1.5·IQR y 3·IQR
# ---------------------------------------------------------------
# Clasificación (Tukey, 1977):
#   • Outlier NO influyente: fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
#   • Outlier INFLUYENTE   : fuera de [Q1 - 3.0·IQR, Q3 + 3.0·IQR]
# Se calcula también el medcouple (MC) como medida robusta de asimetría.

diagn_tukey <- function(x, vname) {
  x    <- x[is.finite(x)]
  q1   <- quantile(x, 0.25, na.rm = TRUE)
  q3   <- quantile(x, 0.75, na.rm = TRUE)
  iqr  <- q3 - q1
  lo15 <- q1 - 1.5 * iqr;  hi15 <- q3 + 1.5 * iqr
  lo3  <- q1 - 3.0 * iqr;  hi3  <- q3 + 3.0 * iqr
  n15  <- sum(x < lo15 | x > hi15)
  n3   <- sum(x < lo3  | x > hi3)
  data.frame(
    Variable    = vname,
    N           = length(x),
    Q1          = round(q1,   1),
    Q3          = round(q3,   1),
    IQR         = round(iqr,  1),
    Lim_inf_1.5 = round(lo15, 1),
    Lim_sup_1.5 = round(hi15, 1),
    Out_1.5_n   = n15,
    Out_1.5_pct = round(100 * n15 / length(x), 1),
    Out_3.0_n   = n3,
    Out_3.0_pct = round(100 * n3  / length(x), 1),
    Asimetria   = round(moments::skewness(x, na.rm = TRUE), 3),
    MC          = round(robustbase::mc(x), 3),
    stringsAsFactors = FALSE
  )
}

diag_tbl <- do.call(rbind,
                    lapply(out_vars, function(v) diagn_tukey(train[[v]], v))
)

cat("\n", strrep("=", 65), "\n")
cat("PASO 6 — DIAGNÓSTICO DE OUTLIERS (Tukey 1.5·IQR y 3·IQR)\n")
cat(strrep("=", 65), "\n")
print(
  diag_tbl %>%
    arrange(desc(Out_1.5_pct)) %>%
    rename(
      `Out 1.5IQR (n)` = Out_1.5_n, `Out 1.5IQR (%)` = Out_1.5_pct,
      `Out 3IQR (n)`   = Out_3.0_n, `Out 3IQR (%)`   = Out_3.0_pct,
      `Asimetría`      = Asimetria
    ),
  row.names = FALSE
)

# ---------------------------------------------------------------
# PASO 7 — Mosaico de boxplots: visualización diagnóstica inicial
# ---------------------------------------------------------------

plots_boxout <- lapply(out_vars, function(v) {
  q1_v  <- quantile(train[[v]], 0.25, na.rm = TRUE)
  q3_v  <- quantile(train[[v]], 0.75, na.rm = TRUE)
  iqr_v <- q3_v - q1_v
  n_out <- sum(train[[v]] < q1_v - 1.5 * iqr_v |
                 train[[v]] > q3_v + 1.5 * iqr_v, na.rm = TRUE)
  ggplot(train, aes(x = factor(1), y = .data[[v]])) +
    geom_boxplot(
      fill = "#4393C3", alpha = 0.70,
      outlier.color = "#D7191C", outlier.alpha = 0.55, outlier.size = 1
    ) +
    scale_y_continuous(labels = label_comma()) +
    labs(
      title    = v,
      subtitle = paste0("Out: ", n_out),
      x = NULL, y = NULL
    ) +
    theme_minimal(base_size = 8) +
    theme(
      plot.title    = element_text(face = "bold", size = 8),
      plot.subtitle = element_text(size = 7, color = "#D7191C"),
      axis.text.x   = element_blank(),
      axis.ticks.x  = element_blank()
    )
})

do.call(grid.arrange, c(plots_boxout, list(
  ncol = 6,
  top  = "PASO 7 — Diagrama de cajas: variables numéricas clave (outliers 1.5·IQR en rojo)"
)))

# ---------------------------------------------------------------
# PASO 8 — Asimetría y boxplot ajustado (Hubert & Vandervieren, 2008)
# ---------------------------------------------------------------
# El boxplot clásico sobreestima outliers en distribuciones asimétricas:
# para variables como LotArea (asimetría ≈ 12), declara outliers legítimos.
# Solución: incorporar el medcouple (MC ∈ [-1, 1]) en los bigotes.
#
# Fórmula (Hubert & Vandervieren, 2008):
#   MC ≥ 0: [Q1 - 1.5·exp(-4·MC)·IQR,  Q3 + 1.5·exp( 3·MC)·IQR]
#   MC < 0: [Q1 - 1.5·exp(-3·MC)·IQR,  Q3 + 1.5·exp( 4·MC)·IQR]
# Con MC = 0 se recupera el boxplot estándar de Tukey.
# Válido para distribuciones unimodales con |MC| ≤ 0.6.

adj_limits_hv <- function(x) {
  x   <- x[is.finite(x)]
  q1  <- quantile(x, 0.25, na.rm = TRUE)
  q3  <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  mc  <- robustbase::mc(x)
  if (mc >= 0) {
    lo <- q1 - 1.5 * exp(-4 * mc) * iqr
    hi <- q3 + 1.5 * exp( 3 * mc) * iqr
  } else {
    lo <- q1 - 1.5 * exp(-3 * mc) * iqr
    hi <- q3 + 1.5 * exp( 4 * mc) * iqr
  }
  list(lower = lo, upper = hi, mc = mc,
       n_out   = sum(x < lo | x > hi),
       pct_out = round(100 * sum(x < lo | x > hi) / length(x), 1))
}

# Tabla comparativa: clásico vs. ajustado para todas las variables
comp_metodos <- do.call(rbind, lapply(out_vars, function(v) {
  al  <- adj_limits_hv(train[[v]])
  row <- diag_tbl[diag_tbl$Variable == v, ]
  data.frame(
    Variable   = v,
    Asimetria  = row$Asimetria,
    MC         = round(al$mc, 3),
    Clas_n     = row$Out_1.5_n,
    Clas_pct   = row$Out_1.5_pct,
    Ajust_n    = al$n_out,
    Ajust_pct  = al$pct_out,
    Falsos_pos = row$Out_1.5_n - al$n_out,
    stringsAsFactors = FALSE
  )
})) %>% arrange(desc(abs(Asimetria)))

cat("\n", strrep("=", 65), "\n")
cat("PASO 8 — CLÁSICO (Tukey 1.5·IQR) vs. AJUSTADO (H&V, 2008)\n")
cat("'Falsos_pos': outliers declarados erróneamente por no corregir asimetría\n")
cat(strrep("=", 65), "\n")
print(
  comp_metodos %>%
    rename(
      `Asimetría`   = Asimetria,
      `Clásico n`   = Clas_n,   `Clásico %`   = Clas_pct,
      `Ajustado n`  = Ajust_n,  `Ajustado %`  = Ajust_pct,
      `Falsos pos.` = Falsos_pos
    ),
  row.names = FALSE
)

# Visualización comparativa: 6 variables con mayor asimetría
vars_asim_top6 <- comp_metodos %>%
  filter(abs(Asimetria) > 1) %>%
  arrange(desc(abs(Asimetria))) %>%
  slice_head(n = 6) %>%
  pull(Variable)

plots_comp_adj <- lapply(vars_asim_top6, function(v) {
  x     <- train[[v]][is.finite(train[[v]])]
  al    <- adj_limits_hv(x)
  q1_v  <- quantile(x, 0.25); q3_v <- quantile(x, 0.75); iqr_v <- q3_v - q1_v
  lo_cl <- q1_v - 1.5 * iqr_v; hi_cl <- q3_v + 1.5 * iqr_v
  
  df_v <- data.frame(x = x) %>%
    mutate(tipo = case_when(
      (x < lo_cl | x > hi_cl) & (x < al$lower | x > al$upper) ~ "Ambos métodos",
      x < lo_cl  | x > hi_cl                                   ~ "Solo clásico",
      x < al$lower | x > al$upper                              ~ "Solo ajustado",
      TRUE                                                       ~ "Normal"
    ))
  
  mc_row <- comp_metodos[comp_metodos$Variable == v, ]
  
  ggplot(df_v, aes(x = 1, y = x)) +
    geom_jitter(data = filter(df_v, tipo == "Normal"),
                alpha = 0.30, size = 0.7, width = 0.2, color = "grey75") +
    geom_jitter(aes(color = tipo), data = filter(df_v, tipo != "Normal"),
                alpha = 0.75, size = 1.3, width = 0.2, show.legend = FALSE) +
    geom_boxplot(fill = NA, color = "black", outlier.shape = NA,
                 width = 0.35, linewidth = 0.6) +
    geom_hline(yintercept = hi_cl,    color = "#D7191C",
               linetype = "dotted",  linewidth = 0.9) +
    geom_hline(yintercept = al$upper, color = "#1A9641",
               linetype = "dashed",  linewidth = 0.9) +
    scale_color_manual(values = c(
      "Ambos métodos" = "#D7191C",
      "Solo clásico"  = "#FDAE61",
      "Solo ajustado" = "#1A9641",
      "Normal"        = "grey75"
    )) +
    scale_y_continuous(labels = label_comma()) +
    labs(
      title    = v,
      subtitle = sprintf("MC = %.2f | Clás: %d | Ajust: %d | Falsos: %d",
                         mc_row$MC, mc_row$Clas_n, mc_row$Ajust_n, mc_row$Falsos_pos),
      x = NULL, y = NULL
    ) +
    theme_minimal(base_size = 8) +
    theme(
      plot.title    = element_text(face = "bold", size = 9),
      plot.subtitle = element_text(size = 6.5),
      axis.text.x   = element_blank()
    )
})

do.call(grid.arrange, c(plots_comp_adj, list(
  ncol = 3,
  top  = paste0(
    "PASO 8 — Clásico (rojo punteado: Q3+1.5·IQR) vs. Ajustado H&V 2008 (verde discontinuo)\n",
    "Rojo = outlier por ambos | Naranja = solo clásico (falso positivo) | Verde = solo ajustado"
  )
)))

# ---------------------------------------------------------------
# PASO 9 — Análisis contextual del dominio inmobiliario
# ---------------------------------------------------------------
# GrLivArea es la variable con outliers más relevantes del dataset.
# Las observaciones con área habitable extrema pero precio inesperadamente
# bajo corresponden a ventas parciales o condiciones atípicas
# (SaleCondition ≠ "Normal"). Sin tratamiento, distorsionan los modelos.

hi_grlivarea_adj <- adj_limits_hv(train$GrLivArea)$upper

train_ctx <- train %>%
  mutate(
    out_grlivarea = GrLivArea > hi_grlivarea_adj,
    SP_fmt  = scales::dollar(SalePrice),
    GrL_fmt = scales::comma(GrLivArea)
  )

p_ctx <- ggplot(train_ctx,
                aes(x = GrLivArea, y = SalePrice, color = out_grlivarea)) +
  geom_point(aes(alpha = out_grlivarea, size = out_grlivarea)) +
  geom_smooth(
    data        = filter(train_ctx, !out_grlivarea),
    mapping     = aes(x = GrLivArea, y = SalePrice),
    inherit.aes = FALSE,
    method = "lm", se = TRUE, color = "black",
    linewidth = 0.8, linetype = "dashed"
  ) +
  geom_vline(xintercept = hi_grlivarea_adj,
             color = "#1A9641", linetype = "dashed", linewidth = 0.8) +
  scale_color_manual(
    values = c("FALSE" = "#4393C3", "TRUE" = "#D7191C"),
    labels = c("Normal", "Outlier (ajustado H&V)")
  ) +
  scale_alpha_manual(values = c("FALSE" = 0.35, "TRUE" = 0.90), guide = "none") +
  scale_size_manual( values = c("FALSE" = 1.2,  "TRUE" = 3.50),  guide = "none") +
  scale_x_continuous(labels = label_comma()) +
  scale_y_continuous(labels = label_dollar()) +
  geom_label_repel(
    data  = filter(train_ctx, out_grlivarea),
    aes(label = paste0("ID:", Id, "\n", GrL_fmt, " sqft\n",
                       SP_fmt, "\n", SaleCondition)),
    size = 2.5, box.padding = 0.4, max.overlaps = 30,
    color = "#D7191C", fill = "white", label.size = 0.2
  ) +
  labs(
    title    = "PASO 9 — Análisis contextual de outliers: GrLivArea vs SalePrice",
    subtitle = paste0(
      "Línea verde: límite superior ajustado H&V (2008)  |  ",
      "Línea negra: tendencia lineal excluyendo outliers"
    ),
    x = "Superficie habitable (sq ft)", y = "Precio de venta (USD)",
    color = NULL
  ) +
  theme_hp +
  theme(legend.position = "bottom")

print(p_ctx)


cat("\n--- Outliers en GrLivArea (método ajustado H&V 2008) ---\n")
cat(sprintf("Límite superior ajustado: %.0f sq ft\n\n", hi_grlivarea_adj))
train %>%
  filter(GrLivArea > hi_grlivarea_adj) %>%
  select(Id, GrLivArea, TotalSF, SalePrice, OverallQual,
         Neighborhood, BldgType, SaleCondition, YearBuilt) %>%
  arrange(desc(GrLivArea)) %>%
  print(n = Inf)

# ---------------------------------------------------------------
# PASO 10 — Decisión estadística y tratamiento: winsorización
# ---------------------------------------------------------------
# La winsorización sustituye valores extremos por el propio percentil
# de corte → conserva la fila, reduce la influencia de extremos.
# Equivale a truncar en el cuantil sin eliminar ninguna observación.
#
# VARIABLES SELECCIONADAS PARA WINSORIZACION [P0.5%, P99.5%]:
#   • LotArea      → Asimetría > 10; lotes de varios acres no representativos
#                    de la distribución central del mercado de Ames
#   • LotFrontage  → Cola derecha pronunciada con valores extremos
#   • GrLivArea    → Outliers documentados (ventas parciales, área > 4.000 sqft)
#   • TotalSF      → Hereda outliers de GrLivArea + TotalBsmtSF
#   • QualSF       → Feature multiplicativa: amplifica outliers de TotalSF
#
# VARIABLES EXCLUIDAS (justificación):
#   • WoodDeckSF, OpenPorchSF, TotalPorchSF → ~40% ceros estructurales;
#     su distribución refleja la ausencia de la característica, no un error
#   • SalePrice   → Valores altos son reales; logSalePrice mitiga su influencia
#   • TotalBsmtSF, GarageArea → Cola moderada y plausible en el dominio
#   • FlrSF_1st, FlrSF_2nd    → Rangos razonables dentro del dataset

winsorize_pct <- function(x, lo_pct = 0.005, hi_pct = 0.995) {
  lo_val <- quantile(x, lo_pct, na.rm = TRUE)
  hi_val <- quantile(x, hi_pct, na.rm = TRUE)
  pmin(pmax(x, lo_val), hi_val)
}

vars_wins <- c("LotArea", "LotFrontage", "GrLivArea", "TotalSF", "QualSF")

# Guardar copia original pre-winsorización (usada en diagnóstico posterior)
train_prewin <- train

# Aplicar winsorización [P0.5%, P99.5%] directamente sobre train
# → A partir de aquí, todo el EDA opera sobre datos saneados
for (v in vars_wins) {
  train[[v]] <- winsorize_pct(train[[v]])
}

cat("\n", strrep("=", 65), "\n")
cat("PASO 10 — WINSORIZACIÓN APLICADA [P0.5%, P99.5%]\n")
cat(strrep("=", 65), "\n")

comp_wins <- do.call(rbind, lapply(vars_wins, function(v) {
  xa <- train_prewin[[v]]
  xd <- train[[v]]
  data.frame(
    Variable      = v,
    Max_antes     = round(max(xa, na.rm = TRUE)),
    Max_despues   = round(max(xd, na.rm = TRUE)),
    SD_antes      = round(sd(xa, na.rm = TRUE), 1),
    SD_despues    = round(sd(xd, na.rm = TRUE), 1),
    Asim_antes    = round(moments::skewness(xa, na.rm = TRUE), 3),
    Asim_despues  = round(moments::skewness(xd, na.rm = TRUE), 3),
    Out15_antes   = sum(xa < quantile(xa, .25) - 1.5 * IQR(xa) |
                          xa > quantile(xa, .75) + 1.5 * IQR(xa), na.rm = TRUE),
    Out15_despues = sum(xd < quantile(xd, .25) - 1.5 * IQR(xd) |
                          xd > quantile(xd, .75) + 1.5 * IQR(xd), na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))

print(
  comp_wins %>%
    rename(
      `Max (antes)`        = Max_antes,   `Max (desp.)`       = Max_despues,
      `SD (antes)`         = SD_antes,    `SD (desp.)`        = SD_despues,
      `Asim. (antes)`      = Asim_antes,  `Asim. (desp.)`     = Asim_despues,
      `Out 1.5IQR (antes)` = Out15_antes, `Out 1.5IQR (desp)` = Out15_despues
    ),
  row.names = FALSE
)

# ---------------------------------------------------------------
# PASO 11 — Verificación visual: comparativa pre/post winsorización
# ---------------------------------------------------------------

# 11a. Boxplots lado a lado por variable
plots_prepost <- lapply(vars_wins, function(v) {
  df_pp <- data.frame(
    Valor = c(train_prewin[[v]], train[[v]]),
    Fase  = rep(c("Antes", "Después"), each = nrow(train))
  ) %>% filter(is.finite(Valor))
  
  ggplot(df_pp, aes(x = Fase, y = Valor, fill = Fase)) +
    geom_boxplot(outlier.alpha = 0.40, outlier.size = 0.8,
                 alpha = 0.80, width = 0.50) +
    scale_fill_manual(values = c("Antes" = "#FC8D59", "Después" = "#74ADD1")) +
    scale_y_continuous(labels = label_comma()) +
    labs(title = v, x = NULL, y = NULL) +
    theme_minimal(base_size = 8) +
    theme(plot.title      = element_text(face = "bold", size = 9),
          legend.position = "none")
})

do.call(grid.arrange, c(plots_prepost, list(
  ncol = 5,
  top  = "PASO 11 — Winsorización [P0.5%, P99.5%]: boxplots antes/después"
)))

# 11b. Curvas de densidad superpuestas: GrLivArea y LotArea
vars_dens_pp <- c("GrLivArea", "LotArea")

plots_dens_pp <- lapply(vars_dens_pp, function(v) {
  df_d <- data.frame(
    Valor = c(train_prewin[[v]], train[[v]]),
    Fase  = rep(c("Original", "Winsorizada [P0.5%, P99.5%]"), each = nrow(train))
  ) %>% filter(is.finite(Valor))
  
  asim_antes <- round(moments::skewness(train_prewin[[v]], na.rm = TRUE), 3)
  asim_desp  <- round(moments::skewness(train[[v]],        na.rm = TRUE), 3)
  
  ggplot(df_d, aes(x = Valor, fill = Fase, color = Fase)) +
    geom_density(alpha = 0.45, linewidth = 0.9) +
    scale_x_continuous(labels = label_comma()) +
    scale_fill_manual( values = c("Original"                     = "#FC8D59",
                                  "Winsorizada [P0.5%, P99.5%]"  = "#74ADD1")) +
    scale_color_manual(values = c("Original"                     = "#D7191C",
                                  "Winsorizada [P0.5%, P99.5%]"  = "#2C7BB6")) +
    labs(
      title    = paste0("Densidad de ", v),
      subtitle = sprintf("Asimetría: %.3f  →  %.3f  (reducción de la cola derecha)",
                         asim_antes, asim_desp),
      x = v, y = "Densidad", fill = NULL, color = NULL
    ) +
    theme_hp +
    theme(legend.position = "bottom")
})

do.call(grid.arrange, c(plots_dens_pp, list(
  ncol = 2,
  top  = "PASO 11 — Efecto de la winsorización sobre la distribución"
)))

# ---------------------------------------------------------------
# PASO 12 — Gráfico interactivo: mapa de outliers (plotly)
# ---------------------------------------------------------------
# Usa train_prewin (valores ORIGINALES) para visualizar la posición
# de los outliers antes del tratamiento. Clasificación bivariante
# mediante el método ajustado H&V en GrLivArea y SalePrice.

hi_adj_grv <- adj_limits_hv(train_prewin$GrLivArea)$upper
hi_adj_sp  <- adj_limits_hv(train_prewin$SalePrice)$upper
lo_adj_sp  <- adj_limits_hv(train_prewin$SalePrice)$lower

df_inter_out <- train_prewin %>%
  mutate(
    OverallQual_n = as.numeric(OverallQual),
    Tipo_outlier  = case_when(
      GrLivArea > hi_adj_grv & SalePrice > hi_adj_sp  ~ "Outlier: área grande + precio alto",
      GrLivArea > hi_adj_grv & SalePrice < lo_adj_sp  ~ "Outlier influyente: área grande, precio bajo",
      GrLivArea > hi_adj_grv                           ~ "Outlier en GrLivArea",
      SalePrice  > hi_adj_sp                           ~ "Outlier en SalePrice",
      TRUE                                             ~ "Normal"
    ),
    SP_fmt  = scales::dollar(SalePrice),
    GrL_fmt = scales::comma(GrLivArea),
    TSF_fmt = scales::comma(TotalSF)
  )

pal_out <- c(
  "Normal"                                       = "#AAAAAA",
  "Outlier en GrLivArea"                         = "#4393C3",
  "Outlier en SalePrice"                         = "#FDB863",
  "Outlier: área grande + precio alto"           = "#D7191C",
  "Outlier influyente: área grande, precio bajo" = "#1A9641"
)

p_inter_out <- plot_ly(
  data      = df_inter_out,
  x         = ~GrLivArea,
  y         = ~SalePrice,
  color     = ~Tipo_outlier,
  colors    = pal_out,
  type      = "scatter",
  mode      = "markers",
  marker    = list(size = 6, opacity = 0.70),
  text      = ~paste0(
    "<b>ID:</b> ",              Id,            "<br>",
    "<b>Precio de venta:</b> ", SP_fmt,        "<br>",
    "<b>Sup. habitable:</b> ",  GrL_fmt,      " sq ft<br>",
    "<b>Sup. total:</b> ",      TSF_fmt,      " sq ft<br>",
    "<b>Calidad general:</b> ", OverallQual,   "<br>",
    "<b>Barrio:</b> ",          Neighborhood,  "<br>",
    "<b>Condición venta:</b> ", SaleCondition, "<br>",
    "<b>Año construido:</b> ",  YearBuilt,     "<br>",
    "<b>Tipo:</b> ",            Tipo_outlier
  ),
  hoverinfo = "text"
) %>%
  layout(
    title = list(
      text = paste0(
        "<b>PASO 12 — Mapa de outliers: SalePrice vs GrLivArea (valores originales)</b>",
        "<br><sup>Método: Boxplot ajustado Hubert & Vandervieren (2008) · ",
        "House Prices Dataset · Ames, Iowa · Kaggle</sup>"
      ),
      font = list(size = 14)
    ),
    xaxis  = list(title = "Superficie habitable (sq ft)", tickformat = ","),
    yaxis  = list(title = "Precio de venta (USD)", tickprefix = "$", tickformat = ","),
    legend = list(
      title       = list(text = "<b>Clasificación del punto</b>"),
      orientation = "h",
      y           = -0.22
    ),
    hovermode     = "closest",
    plot_bgcolor  = "#F8F9FA",
    paper_bgcolor = "#FFFFFF",
    font          = list(family = "Arial")
  )

print(p_inter_out)

# ---------------------------------------------------------------
# Resumen del tratamiento de outliers (a.4)
# ---------------------------------------------------------------
cat("\n", strrep("=", 65), "\n")
cat("RESUMEN — TRATAMIENTO DE OUTLIERS (a.4)\n")
cat(strrep("=", 65), "\n")
cat(sprintf("Variables diagnosticadas              : %d\n",  length(out_vars)))
cat(sprintf("Variables con |Asimetría| > 1         : %d\n",  sum(abs(diag_tbl$Asimetria) > 1)))
cat(sprintf("Variables con falsos positivos clás.  : %d\n",  sum(comp_metodos$Falsos_pos > 0)))
cat(sprintf("Variables winsorizadas [P0.5, P99.5]  : %d\n",  length(vars_wins)))
cat(sprintf("  → %s\n",                                       paste(vars_wins, collapse = ", ")))
cat(sprintf("Observaciones (sin eliminar ninguna)  : %d\n",  nrow(train)))
cat(sprintf("train actualizado · train_prewin conservado como copia original\n"))
cat("\nMetodología:\n")
cat("  PASO 6  — Diagnóstico Tukey 1.5·IQR (no influyentes) y 3·IQR (influyentes)\n")
cat("  PASO 7  — Mosaico de boxplots diagnósticos\n")
cat("  PASO 8  — Boxplot ajustado H&V (2008) con medcouple\n")
cat("            → Corrección del sesgo; evita falsos positivos por asimetría\n")
cat("  PASO 9  — Análisis contextual del dominio inmobiliario\n")
cat("  PASO 10 — Winsorización conservadora [P0.5%, P99.5%]\n")
cat("  PASO 11 — Verificación visual pre/post (boxplots + densidades)\n")
cat("  PASO 12 — Gráfico interactivo: mapa de outliers\n")
cat(strrep("=", 65), "\n")




# ==============================================================
# b) ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================
# ... (código existente sin tocar — opera sobre train ya saneado)



# ==============================================================
# b) ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================

# ==============================================================
# b.1) ESTUDIO DESCRIPTIVO
# ==============================================================

# ------------------------------------------------------------------
# BLOQUE 1 — Distribución de la variable respuesta: SalePrice
# ------------------------------------------------------------------
p_sp1 <- ggplot(train, aes(x = SalePrice)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50,
                 fill = "#2C7BB6", alpha = 0.75, color = "white") +
  geom_density(color = "#D7191C", linewidth = 1.1) +
  scale_x_continuous(labels = label_dollar()) +
  labs(
    title    = "Distribución de SalePrice (precio de venta)",
    subtitle = sprintf("n = %d  |  Media = %s  |  Mediana = %s  |  Asimetría = %.2f  |  Curtosis = %.2f",
                       nrow(train),
                       dollar(round(mean(train$SalePrice))),
                       dollar(median(train$SalePrice)),
                       skewness(train$SalePrice),
                       kurtosis(train$SalePrice)),
    x = "Precio de venta (USD)",
    y = "Densidad"
  ) + theme_hp

p_sp2 <- ggplot(train, aes(x = logSalePrice)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50,
                 fill = "#1A9641", alpha = 0.75, color = "white") +
  geom_density(color = "#D7191C", linewidth = 1.1) +
  labs(
    title    = "Distribución de log(SalePrice)",
    subtitle = sprintf("La transformación logarítmica normaliza la distribución  |  Asimetría = %.2f  |  Curtosis = %.2f",
                       skewness(train$logSalePrice),
                       kurtosis(train$logSalePrice)),
    x = "log(Precio de venta)",
    y = "Densidad"
  ) + theme_hp

grid.arrange(p_sp1, p_sp2, ncol = 2,
             top = "Variable respuesta: SalePrice y su transformación logarítmica")


# ------------------------------------------------------------------
# BLOQUE 2 — Tabla de estadísticos descriptivos (vars. numéricas)
# ------------------------------------------------------------------
num_vars_key <- c(
  "SalePrice", "LotArea", "LotFrontage", "GrLivArea",
  "TotalSF",   "TotalBsmtSF", "FlrSF_1st", "FlrSF_2nd",
  "GarageArea","TotalBaths",  "HouseAge",  "RemodAge",
  "Fireplaces","TotRmsAbvGrd","WoodDeckSF","OpenPorchSF"
)

stats_tbl <- train %>%
  select(all_of(num_vars_key)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Valor") %>%
  group_by(Variable) %>%
  summarise(
    N       = sum(!is.na(Valor)),
    Media   = round(mean(Valor,                  na.rm = TRUE), 1),
    Mediana = round(median(Valor,                na.rm = TRUE), 1),
    DT      = round(sd(Valor,                    na.rm = TRUE), 1),
    Min     = round(min(Valor,                   na.rm = TRUE), 1),
    Q1      = round(quantile(Valor, 0.25,        na.rm = TRUE), 1),
    Q3      = round(quantile(Valor, 0.75,        na.rm = TRUE), 1),
    Max     = round(max(Valor,                   na.rm = TRUE), 1),
    Asim    = round(skewness(Valor,              na.rm = TRUE), 2),
    Kurt    = round(kurtosis(Valor,              na.rm = TRUE), 2),
    .groups = "drop"
  ) %>%
  arrange(desc(abs(Asim)))

cat("\nTabla de estadísticos descriptivos — variables numéricas clave:\n")
print(stats_tbl, n = Inf)


# ------------------------------------------------------------------
# BLOQUE 3 — Histogramas en mosaico (variables numéricas clave)
# ------------------------------------------------------------------
hist_vars <- c(
  "LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea",
  "TotalSF", "TotalBaths", "HouseAge",   "LotFrontage"
)

plots_hist <- lapply(hist_vars, function(v) {
  ggplot(train, aes(x = .data[[v]])) +
    geom_histogram(bins = 40, fill = "#4393C3", alpha = 0.85, color = "white") +
    scale_x_continuous(labels = label_comma()) +
    labs(title = v, x = NULL, y = "Frec.") +
    theme_minimal(base_size = 8) +
    theme(plot.title = element_text(face = "bold", size = 9))
})

do.call(grid.arrange, c(plots_hist, ncol = 4,
                        top = "Distribución de variables numéricas clave"))


# ------------------------------------------------------------------
# BLOQUE 4 — Boxplot: SalePrice por calidad general (OverallQual)
# ------------------------------------------------------------------
p_qual <- ggplot(train, aes(x = OverallQual, y = SalePrice, fill = OverallQual)) +
  geom_boxplot(outlier.alpha = 0.35, outlier.size = 0.8) +
  scale_y_continuous(labels = label_dollar()) +
  scale_fill_viridis_d(option = "plasma") +
  labs(
    title    = "Precio de venta por Calidad General (OverallQual)",
    subtitle = "Escala ordinal: 1 = Muy deficiente → 10 = Excelente",
    x = "Calidad general",
    y = "Precio de venta (USD)"
  ) +
  theme_hp +
  theme(legend.position = "none")
print(p_qual)


# ------------------------------------------------------------------
# BLOQUE 5 — Scatterplot: SalePrice vs GrLivArea
# ------------------------------------------------------------------
p_scatter1 <- ggplot(train,
                     aes(x = GrLivArea, y = SalePrice,
                         color = as.numeric(OverallQual))) +
  geom_point(alpha = 0.45, size = 1.5) +
  geom_smooth(method = "lm", color = "black",
              se = TRUE, linewidth = 0.9, linetype = "dashed") +
  scale_y_continuous(labels = label_dollar()) +
  scale_x_continuous(labels = label_comma()) +
  scale_color_viridis_c(option = "plasma", name = "Calidad\nGeneral") +
  labs(
    title    = "SalePrice vs Superficie habitable (GrLivArea)",
    subtitle = sprintf("Correlación de Pearson: r = %.3f",
                       cor(train$SalePrice, train$GrLivArea)),
    x = "Superficie habitable (sq ft)",
    y = "Precio de venta (USD)"
  ) + theme_hp
print(p_scatter1)


# ------------------------------------------------------------------
# BLOQUE 6 — Scatterplot: SalePrice vs TotalSF (variable creada)
# ------------------------------------------------------------------
p_scatter2 <- ggplot(train,
                     aes(x = TotalSF, y = SalePrice,
                         color = as.numeric(OverallQual))) +
  geom_point(alpha = 0.45, size = 1.5) +
  geom_smooth(method = "lm", color = "black",
              se = TRUE, linewidth = 0.9, linetype = "dashed") +
  scale_y_continuous(labels = label_dollar()) +
  scale_x_continuous(labels = label_comma()) +
  scale_color_viridis_c(option = "plasma", name = "Calidad\nGeneral") +
  labs(
    title    = "SalePrice vs Superficie Total (TotalSF = sótano + P.baja + P.alta)",
    subtitle = sprintf("Correlación de Pearson: r = %.3f",
                       cor(train$SalePrice, train$TotalSF)),
    x = "Superficie total (sq ft)",
    y = "Precio de venta (USD)"
  ) + theme_hp
print(p_scatter2)


# ------------------------------------------------------------------
# BLOQUE 7 — Precio mediano por Neighborhood (barras horizontales)
# ------------------------------------------------------------------
p_neigh <- train %>%
  group_by(Neighborhood) %>%
  summarise(median_price = median(SalePrice), n = n(), .groups = "drop") %>%
  mutate(Neighborhood = fct_reorder(Neighborhood, median_price)) %>%
  ggplot(aes(x = Neighborhood, y = median_price, fill = median_price)) +
  geom_col() +
  geom_text(aes(label = paste0("n=", n)),
            hjust = -0.1, size = 2.7, color = "grey30") +
  scale_y_continuous(
    labels = label_dollar(),
    expand = expansion(mult = c(0, 0.18))
  ) +
  scale_fill_viridis_c(option = "magma", labels = label_dollar(),
                       name = "Precio\nmediano") +
  coord_flip() +
  labs(
    title    = "Precio mediano de venta por Barrio (Neighborhood)",
    subtitle = "Ordenado de menor a mayor precio mediano",
    x = NULL, y = "Precio mediano (USD)"
  ) + theme_hp
print(p_neigh)


# ------------------------------------------------------------------
# BLOQUE 8 — Distribución de precios por año (ridgeline plot)
# ------------------------------------------------------------------
p_ridge <- ggplot(train,
                  aes(x = SalePrice, y = YrSold, fill = YrSold)) +
  geom_density_ridges(
    alpha          = 0.70,
    scale          = 1.10,
    quantile_lines = TRUE,
    quantiles      = c(0.25, 0.5, 0.75)
  ) +
  scale_x_continuous(labels = label_dollar()) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title    = "Distribución de precios de venta por año",
    subtitle = "Las líneas verticales marcan Q1, mediana y Q3",
    x = "Precio de venta (USD)",
    y = "Año de venta"
  ) +
  theme_hp +
  theme(legend.position = "none")
print(p_ridge)


# ------------------------------------------------------------------
# BLOQUE 9 — Precio mediano y volumen de ventas por mes
# ------------------------------------------------------------------
p_mes <- train %>%
  group_by(MoSold) %>%
  summarise(median_price = median(SalePrice), n = n(), .groups = "drop") %>%
  ggplot(aes(x = MoSold, y = median_price, group = 1)) +
  geom_line(color = "#2C7BB6", linewidth = 1.2) +
  geom_point(aes(size = n), color = "#D7191C", alpha = 0.85) +
  scale_y_continuous(labels = label_dollar()) +
  labs(
    title    = "Precio mediano y volumen de ventas por mes",
    subtitle = "El tamaño del punto indica el número de transacciones",
    x = "Mes de venta",
    y = "Precio mediano (USD)",
    size = "Nº ventas"
  ) + theme_hp
print(p_mes)


# ------------------------------------------------------------------
# BLOQUE 10 — Variables indicadoras binarias (HasXxx)
# ------------------------------------------------------------------
bin_vars <- c("HasPool", "HasGarage", "HasFireplace", "HasBasement")

plots_bin <- lapply(bin_vars, function(v) {
  train %>%
    group_by(.data[[v]]) %>%
    summarise(mediana = median(SalePrice), n = n(), .groups = "drop") %>%
    ggplot(aes(x = .data[[v]], y = mediana, fill = .data[[v]])) +
    geom_col(width = 0.55) +
    geom_text(aes(label = paste0("n=", n)),
              vjust = -0.4, size = 3, color = "grey30") +
    scale_y_continuous(labels  = label_dollar(),
                       expand  = expansion(mult = c(0, 0.15))) +
    scale_fill_manual(values = c("No" = "#FC8D59", "Sí" = "#74ADD1")) +
    labs(title = v, x = NULL, y = "Precio mediano") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "none")
})

do.call(grid.arrange, c(plots_bin, ncol = 4,
                        top = "Precio mediano según características adicionales de la vivienda"))


# ------------------------------------------------------------------
# BLOQUE 11 — Violin + Boxplot por tipo de edificio (BldgType)
# ------------------------------------------------------------------
p_bldg <- ggplot(train,
                 aes(x = BldgType, y = SalePrice, fill = BldgType)) +
  geom_violin(trim = FALSE, alpha = 0.65) +
  geom_boxplot(width = 0.1, fill = "white",
               outlier.size = 0.5, outlier.alpha = 0.4) +
  scale_y_continuous(labels = label_dollar()) +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title = "Distribución del precio de venta por tipo de edificio (BldgType)",
    x = "Tipo de edificio",
    y = "Precio de venta (USD)"
  ) +
  theme_hp +
  theme(legend.position = "none")
print(p_bldg)


# ------------------------------------------------------------------
# BLOQUE 12 — Frecuencias de variables categóricas clave
# ------------------------------------------------------------------
cat_key <- c(
  "MSZoning",     "BldgType",    "HouseStyle",
  "SaleCondition","Foundation",  "GarageType",
  "CentralAir",   "Neighborhood"
)

plots_bar <- lapply(cat_key, function(v) {
  tmp <- train %>%
    count(.data[[v]], name = "n") %>%
    arrange(desc(n)) %>%
    slice_head(n = 15) %>%
    mutate(nivel = fct_reorder(as.character(.data[[v]]), n))
  
  ggplot(tmp, aes(x = nivel, y = n)) +
    geom_col(fill = "#2CA25F", alpha = 0.85) +
    coord_flip() +
    labs(title = v, x = NULL, y = "Frec.") +
    theme_minimal(base_size = 8) +
    theme(plot.title = element_text(face = "bold", size = 8))
})

do.call(grid.arrange, c(plots_bar, ncol = 4,
                        top = "Frecuencias de variables categóricas clave"))


# ------------------------------------------------------------------
# BLOQUE 13 — Tabla de frecuencias: OverallQual
# ------------------------------------------------------------------
freq_qual <- train %>%
  count(OverallQual, name = "Frecuencia") %>%
  mutate(
    `Porcentaje (%)`     = round(100 * Frecuencia / sum(Frecuencia), 1),
    `Frec. acumulada`    = cumsum(Frecuencia),
    `Porc. acumulado (%)` = round(100 * cumsum(Frecuencia) / sum(Frecuencia), 1)
  ) %>%
  rename(`Calidad general` = OverallQual)

cat("\nTabla de frecuencias: OverallQual\n")
print(freq_qual)


# ------------------------------------------------------------------
# BLOQUE 14 — Tabla resumen: SalePrice por OverallQual
# ------------------------------------------------------------------
tab_qual <- train %>%
  group_by(OverallQual) %>%
  summarise(
    N              = n(),
    Mínimo         = dollar(min(SalePrice)),
    Q1             = dollar(quantile(SalePrice, 0.25)),
    Mediana        = dollar(median(SalePrice)),
    Media          = dollar(round(mean(SalePrice))),
    Q3             = dollar(quantile(SalePrice, 0.75)),
    Máximo         = dollar(max(SalePrice)),
    DT             = dollar(round(sd(SalePrice))),
    .groups = "drop"
  )

cat("\nEstadísticos de SalePrice por Calidad General (OverallQual):\n")
print(tab_qual)


# ------------------------------------------------------------------
# BLOQUE 15 — Matriz de correlación (vars. numéricas)
# ------------------------------------------------------------------
cor_vars <- c(
  "SalePrice",  "GrLivArea",   "LotArea",    "LotFrontage",
  "TotalBsmtSF","GarageArea",  "TotalSF",    "TotalBaths",
  "HouseAge",   "RemodAge",    "TotRmsAbvGrd","Fireplaces",
  "WoodDeckSF", "OpenPorchSF", "OverallQual_n"
)

train_cor <- train %>%
  mutate(OverallQual_n = as.numeric(OverallQual)) %>%
  select(all_of(cor_vars))

cor_mat <- cor(train_cor, use = "pairwise.complete.obs")

par(mar = c(0, 0, 3, 0))
corrplot(
  cor_mat,
  method      = "color",
  type        = "upper",
  order       = "hclust",
  tl.cex      = 0.78,
  tl.col      = "black",
  addCoef.col = "black",
  number.cex  = 0.52,
  col         = colorRampPalette(c("#D7191C", "white", "#2C7BB6"))(200),
  title       = "Matriz de correlación — variables numéricas clave",
  mar         = c(0, 0, 2, 0)
)



# ------------------------------------------------------------------
# BLOQUE 15b — Matriz de correlación (extendida con QualSF)
# ------------------------------------------------------------------

cor_vars_reina <- c(cor_vars, "QualSF")

train_cor_reina <- train %>%
  mutate(OverallQual_n = as.numeric(OverallQual)) %>%
  select(all_of(cor_vars_reina))

cor_mat_reina <- cor(train_cor_reina, use = "pairwise.complete.obs")

par(mar = c(0, 0, 3, 0))
corrplot(
  cor_mat_reina,
  method = "color",
  type = "upper",
  order = "hclust",
  tl.cex = 0.78,
  tl.col = "black",
  addCoef.col = "black",
  number.cex = 0.52,
  col = colorRampPalette(c("#D7191C", "white", "#2C7BB6"))(200),
  title = "Matriz de correlación — variables numéricas clave (+ QualSF)",
  mar = c(0, 0, 2, 0)
)



# ------------------------------------------------------------------
# BLOQUE 16 — Top 10 correlaciones con SalePrice
# ------------------------------------------------------------------
cor_sp    <- cor_mat["SalePrice", ]
top_cor   <- sort(abs(cor_sp[names(cor_sp) != "SalePrice"]), decreasing = TRUE)[1:10]

p_cor <- data.frame(
  Variable    = names(top_cor),
  Correlacion = as.numeric(top_cor)
) %>%
  mutate(Variable = fct_reorder(Variable, Correlacion)) %>%
  ggplot(aes(x = Variable, y = Correlacion, fill = Correlacion)) +
  geom_col() +
  geom_text(aes(label = round(Correlacion, 3)),
            hjust = -0.1, size = 3.2) +
  scale_fill_gradient(low = "#FEE090", high = "#D7191C") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  coord_flip() +
  labs(
    title    = "Top 10: Variables con mayor correlación con SalePrice",
    subtitle = "Valor absoluto de la correlación de Pearson",
    x = NULL, y = "|r de Pearson|",
    fill = "|r|"
  ) + theme_hp
print(p_cor)


# meter reina pepiada, y cambiar lo de pearsons



# ==============================================================
# b.2) GRÁFICO INTERACTIVO (plotly)
# ==============================================================
# Scatter interactivo: SalePrice vs TotalSF
# Coloreado por OverallQual — Hover con info detallada por vivienda

df_inter <- train %>%
  mutate(
    OverallQual_n = as.numeric(OverallQual),
    SP_fmt        = scales::dollar(SalePrice),
    SF_fmt        = scales::comma(TotalSF)
  )

p_interactive <- plot_ly(
  data      = df_inter,
  x         = ~TotalSF,
  y         = ~SalePrice,
  color     = ~OverallQual_n,
  colors    = viridis(10, option = "plasma"),
  type      = "scatter",
  mode      = "markers",
  marker    = list(size = 6, opacity = 0.65),
  text      = ~paste0(
    "<b>Precio de venta:</b> ", SP_fmt,
    "<br><b>Superficie total:</b> ", SF_fmt, " sq ft",
    "<br><b>Barrio:</b> ", Neighborhood,
    "<br><b>Calidad general:</b> ", OverallQual,
    "<br><b>Tipo de edificio:</b> ", BldgType,
    "<br><b>Estilo de la casa:</b> ", HouseStyle,
    "<br><b>Año construido:</b> ", YearBuilt,
    "<br><b>Año vendido:</b> ", YrSold,
    "<br><b>Condición de venta:</b> ", SaleCondition
  ),
  hoverinfo = "text"
) %>%
  layout(
    title = list(
      text = paste0(
        "<b>Precio de venta vs Superficie Total</b><br>",
        "<sup>Coloreado por Calidad General (1–10) — ",
        "House Prices Dataset · Ames, Iowa · Kaggle</sup>"
      ),
      font = list(size = 15)
    ),
    xaxis = list(
      title      = "Superficie total (sq ft)",
      tickformat = ","
    ),
    yaxis = list(
      title      = "Precio de venta (USD)",
      tickprefix = "$",
      tickformat = ","
    ),
    coloraxis = list(
      colorbar = list(
        title    = "<b>Calidad<br>General</b>",
        tickvals = 1:10
      )
    ),
    hovermode     = "closest",
    plot_bgcolor  = "#F8F9FA",
    paper_bgcolor = "#FFFFFF",
    font          = list(family = "Arial")
  )

print(p_interactive)


# ==============================================================
# b.2b) GRÁFICO INTERACTIVO (plotly) — SalePrice vs QualSF
# ==============================================================

df_inter_reina <- train %>%
  mutate(
    OverallQual_n = as.numeric(OverallQual),
    SP_fmt = scales::dollar(SalePrice),
    QualSF_fmt = scales::comma(QualSF),
    TotalSF_fmt = scales::comma(TotalSF)
  )

p_interactive_reina <- plot_ly(
  data = df_inter_reina,
  x = ~QualSF,
  y = ~SalePrice,
  color = ~OverallQual_n,
  colors = viridis(10, option = "plasma"),
  type = "scatter",
  mode = "markers",
  marker = list(size = 6, opacity = 0.65),
  text = ~paste0(
    "Precio de venta: ", SP_fmt, "<br>",
    "Feature reina (QualSF): ", QualSF_fmt, "<br>",
    "Superficie total (TotalSF): ", TotalSF_fmt, " sq ft", "<br>",
    "Barrio: ", Neighborhood, "<br>",
    "Calidad general: ", OverallQual, "<br>",
    "Tipo de edificio: ", BldgType, "<br>",
    "Estilo de la casa: ", HouseStyle, "<br>",
    "Año construido: ", YearBuilt, "<br>",
    "Año vendido: ", YrSold, "<br>",
    "Condición de venta: ", SaleCondition
  ),
  hoverinfo = "text"
) %>%
  layout(
    title = list(
      text = paste0(
        "Precio de venta vs Feature reina (QualSF = OverallQual × TotalSF)",
        "<br><sup>Coloreado por Calidad General (1–10) — House Prices Dataset · Ames, Iowa · Kaggle</sup>"
      ),
      font = list(size = 15)
    ),
    xaxis = list(
      title = "QualSF (OverallQual × TotalSF)",
      tickformat = ","
    ),
    yaxis = list(
      title = "Precio de venta (USD)",
      tickprefix = "$",
      tickformat = ","
    ),
    coloraxis = list(
      colorbar = list(
        title = "Calidad<br>General",
        tickvals = 1:10
      )
    ),
    hovermode = "closest",
    plot_bgcolor = "#F8F9FA",
    paper_bgcolor = "#FFFFFF",
    font = list(family = "Arial")
  )

print(p_interactive_reina)



# ==============================================================
cat("\n", strrep("=", 65), "\n")
cat("✓ Análisis completado con éxito.\n")
cat(strrep("=", 65), "\n")

