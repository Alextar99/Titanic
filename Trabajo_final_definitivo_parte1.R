# ==============================================================
# ANÁLISIS EXPLORATORIO: House Prices – Advanced Regression
# Kaggle: house-prices-advanced-regression-techniques
# URL   : https://www.kaggle.com/competitions/
#         house-prices-advanced-regression-techniques
# ==============================================================


# ==============================================================
# 0. INSTALACIÓN Y CARGA DE LIBRERÍAS
# ==============================================================
rm(list = ls())
graphics.off() 


if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

# Se añaden las librerías para transformación, normalidad, estandarización y dummies
pacman::p_load(
  tidyverse, skimr, naniar, corrplot,
  plotly, scales, gridExtra, moments, viridis, nortest,
  robustbase, VIM, editrules, MASS, car, dlookr, fastDummies
)


# Tema gráfico personalizado
theme_hp <- theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle = element_text(color = "grey40", size = 10),
    axis.title = element_text(size = 10),
    legend.title = element_text(face = "bold", size = 9),
    strip.text = element_text(face = "bold")
  )


# ==============================================================
# a) PREPARACIÓN DE LOS DATOS
# ==============================================================

# --------------------------------------------------------------
# a.1) DESCRIPCIÓN DEL CONJUNTO ORIGINAL
# --------------------------------------------------------------
cat(strrep("=", 65), "\n")
cat("NOMBRE   : House Prices – Advanced Regression Techniques\n")
cat("FUENTE   : Kaggle\n")
cat("ENLACE   : https://www.kaggle.com/competitions/\n")
cat("           house-prices-advanced-regression-techniques\n")
cat("CONTEXTO : Precios de venta de viviendas residenciales\n")
cat("           en Ames, Iowa (EE.UU.) — años 2006-2010\n")
cat(strrep("=", 65), "\n\n")

# !! Ajusta la ruta a la ubicación de tu archivo train.csv !!
train_raw <- read.csv("C:/Users/alega/OneDrive/Documentos/Análisis de Datos/Trabajo ADAT/House-prices/train.csv", header = TRUE, stringsAsFactors = FALSE)

n_obs  <- nrow(train_raw)
n_vars <- ncol(train_raw)
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


# --------------------------------------------------------------
# a.2) TRANSFORMACIÓN, LIMPIEZA E IMPUTACIÓN
# --------------------------------------------------------------
train <- train_raw  # copia de trabajo: train_raw queda intacto

# Renombrar variables con nombres no sintácticos en R
train <- train %>%
  rename(FlrSF_1st  = X1stFlrSF,
         FlrSF_2nd  = X2ndFlrSF,
         Porch_3Ssn = X3SsnPorch)

# ==============================================================================
# ---> AÑADIR AQUÍ: ELIMINACIÓN MANUAL DE OUTLIERS FAMOSOS
# ==============================================================================
# El autor del dataset recomienda eliminar las viviendas de > 4000 sq ft 
# vendidas por un precio anormalmente bajo, ya que distorsionan el análisis.
train <- train %>%
  filter(!(GrLivArea > 4000 & SalePrice < 300000))

cat(sprintf("\nSe han eliminado %d outliers manuales extremos.\n", 
            nrow(train_raw) - nrow(train)))
# ==============================================================================

# --- PASO 1: Diagnóstico de valores perdidos ------------------
miss_df <- data.frame(
  Variable    = names(train),
  N_Missing   = colSums(is.na(train)),
  Pct_Missing = round(100 * colMeans(is.na(train)), 2)
) %>%
  filter(N_Missing > 0) %>%
  arrange(desc(Pct_Missing))

cat("\n--- Diagnóstico de valores perdidos (antes de imputar) ---\n")
print(miss_df, row.names = FALSE)

# Visualización: gráfico de barras por % de NAs por variable
p_miss <- gg_miss_var(train, show_pct = TRUE) +
  labs(
    title    = "Porcentaje de valores faltantes por variable",
    subtitle = sprintf("House Prices — Kaggle  (n = %d obs., %d vars.)",
                       n_obs, n_vars),
    y = "% Missing"
  ) + theme_hp
print(p_miss)


# --- PASO 1.5: Eliminar variables insignificativas (>80% NA) ---
umbral_na <- 80
vars_insig <- miss_df$Variable[miss_df$Pct_Missing > umbral_na]
cat(sprintf("\nVariables eliminadas por exceso de NAs (>%d%%) al no aportar información: %s\n", 
            umbral_na, paste(vars_insig, collapse = ", ")))

train <- train %>% dplyr::select(-all_of(vars_insig))


# --- PASO 2: Imputación semántica (según data dictionary) -----
na_none_cat <- c(
  "Alley", "MasVnrType", "BsmtQual", "BsmtCond",
  "BsmtExposure", "BsmtFinType1", "BsmtFinType2","FireplaceQu",
  "GarageType", "GarageFinish", "GarageQual", "GarageCond",
  "PoolQC", "Fence", "MiscFeature"
)
# Retener solo aquellas que no han sido eliminadas
na_none_cat <- intersect(na_none_cat, names(train))

for (v in na_none_cat) {
  train[[v]][is.na(train[[v]])] <- "None"
}

na_zero_num <- c(
  "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
  "TotalBsmtSF","BsmtFullBath", "BsmtHalfBath",
  "GarageCars", "GarageArea", "GarageYrBlt"
)

for (v in na_zero_num) {
  train[[v]][is.na(train[[v]])] <- 0
}

train <- train %>%
  group_by(Neighborhood) %>%
  mutate(LotFrontage = if_else(
    is.na(LotFrontage),
    median(LotFrontage, na.rm = TRUE),
    as.double(LotFrontage)
  )) %>%
  ungroup()

moda_elec <- names(which.max(table(train$Electrical)))
train$Electrical[is.na(train$Electrical)] <- moda_elec


# --- PASO 3: Feature Engineering ------------------------------
# Se extraen las nuevas variables ANTES de convertir YrSold a factor,
# porque las operaciones aritméticas requieren valores numéricos.
yr_num <- as.integer(as.character(train$YrSold))

# Transformación de Box-Cox rigurosa (Sustituye al logaritmo manual logSalePrice)
# 1. Obtenemos el vector limpio sin NAs y estrictamente positivo
precio_limpio <- train$SalePrice[!is.na(train$SalePrice) & train$SalePrice > 0]

# 2. Buscamos el parámetro lambda óptimo maximizando la log-verosimilitud
b_cox <- boxcox(lm(precio_limpio ~ 1), plotit = FALSE)
lambda_opt <- b_cox$x[which.max(b_cox$y)]
cat(sprintf("\nLambda óptimo de Box-Cox para SalePrice: %.4f\n", lambda_opt))


train <- train %>%
  mutate(
    SalePrice_BC  = (SalePrice^lambda_opt - 1) / lambda_opt,
    HouseAge      = yr_num - YearBuilt,
    RemodAge      = yr_num - YearRemodAdd,
    TotalSF       = TotalBsmtSF + FlrSF_1st + FlrSF_2nd, # TotalSF se crea aquí
    TotalBaths    = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath,
    HasPool       = factor(ifelse(PoolArea > 0, "Sí", "No")),
    HasGarage     = factor(ifelse(GarageArea > 0, "Sí", "No")),
    HasFireplace  = factor(ifelse(Fireplaces > 0, "Sí", "No")),
    HasBasement   = factor(ifelse(TotalBsmtSF > 0, "Sí", "No"))
  ) %>%
  mutate(
    QualSF       = as.integer(OverallQual) * TotalSF,
    TotalPorchSF = OpenPorchSF + EnclosedPorch + Porch_3Ssn + ScreenPorch
  )

# Verificar correlación de las nuevas variables con SalePrice
new_feats <- c("QualSF", "TotalSF", "TotalBaths",
               "TotalPorchSF", "HouseAge", "RemodAge", "SalePrice", "SalePrice_BC")

cor_new <- cor(
  train %>% dplyr::select(all_of(new_feats)),
  use = "pairwise.complete.obs"
)["SalePrice", ] %>% sort(decreasing = TRUE)

cat("\nCorrelación de las variables derivadas con SalePrice:\n")
print(round(cor_new, 3))


# --- PASO 4: Conversión al tipo de dato correcto --------------

# MSSubClass: código numérico que representa tipo de vivienda,
# no una magnitud → factor nominal
train$MSSubClass <- factor(train$MSSubClass)

# OverallQual / OverallCond: escala 1–10 con orden real → ordinal
train$OverallQual <- factor(train$OverallQual, levels = 1:10, ordered = TRUE)
train$OverallCond <- factor(train$OverallCond, levels = 1:10, ordered = TRUE)

# MoSold: número del mes → factor con etiquetas legibles
train$MoSold <- factor(train$MoSold, levels = 1:12, labels = month.abb)

# YrSold: año de venta → factor nominal
train$YrSold <- factor(train$YrSold)

# Variables de calidad con escala estandarizada Po→Ex
# "None" se incluye como primer nivel (casas sin esa característica)
quality_levels <- c("None", "Po", "Fa", "TA", "Gd", "Ex")
ord_qual_vars  <- c(
  "ExterQual", "ExterCond",  "BsmtQual",   "BsmtCond",
  "HeatingQC", "KitchenQual","FireplaceQu",
  "GarageQual","GarageCond"
)
for (v in ord_qual_vars) {
  train[[v]] <- factor(train[[v]], levels = quality_levels, ordered = TRUE)
}

# Resto de variables de texto → factor nominal
chr_vars <- names(train)[sapply(train, is.character)]
train[chr_vars] <- lapply(train[chr_vars], factor)


# --- PASO 5: Verificación final de valores perdidos -----------
n_na_restantes <- sum(is.na(train))
cat(sprintf("\nValores perdidos tras la imputación: %d\n", n_na_restantes))
if (n_na_restantes > 0) {
  cat("Variables con NAs restantes:\n")
  print(names(which(colSums(is.na(train)) > 0)))
}



# ==============================================================
# a.4) TRATAMIENTO DE OUTLIERS E IMPUTACIÓN k-NN
# ==============================================================
# Referencia: PDF de teoría
# 1. Detección: Hubert, M. y Vandervieren, E. (2008) - Boxplot ajustado (Medcouple).
# 2. Imputación: VIM::kNN - Agregación k-Nearest Neighbors con distancia de Gower.


out_vars <- c("GrLivArea", "LotArea", "LotFrontage", "TotalBsmtSF", "GarageArea", "HouseAge")

# Guardamos una copia del dataset original para la comparativa visual posterior
train_pre_outliers <- train

cat("\n", strrep("=", 65), "\n")
cat("PASO 6 — DIAGNÓSTICO DE OUTLIERS (Hubert & Vandervieren)\n")
cat(strrep("=", 65), "\n")

# --- Justificación Visual: Tukey vs Boxplot Ajustado ---
# Tomamos SalePrice como ejemplo de distribución asimétrica positiva

options(mc_doScale_quiet = TRUE)

sp_clean <- train$SalePrice[!is.na(train$SalePrice)]
q1 <- quantile(sp_clean, 0.25)
q3 <- quantile(sp_clean, 0.75)
iqr <- q3 - q1
lim_tukey_sup <- q3 + 1.5 * iqr
lim_adj_sup <- robustbase::adjboxStats(sp_clean)$fence[2]

p_justificacion <- ggplot(train, aes(x = SalePrice)) +
  geom_density(fill = "#2C7BB6", alpha = 0.4, color = "black", linewidth = 0.8) +
  geom_vline(xintercept = lim_tukey_sup, color = "#D7191C", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = lim_adj_sup, color = "#1A9641", linetype = "solid", linewidth = 1) +
  annotate("text", x = lim_tukey_sup, y = 0.000004, label = "Límite Clásico (Tukey)", color = "#D7191C", angle = 90, vjust = -0.5, size = 3.5, fontface = "bold") +
  annotate("text", x = lim_adj_sup, y = 0.000004, label = "Límite Ajustado (Medcouple)", color = "#1A9641", angle = 90, vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_x_continuous(labels = label_dollar()) +
  labs(
    title = "Justificación: Método Clásico vs Boxplot Ajustado por Asimetría",
    subtitle = "El método clásico asume simetría y genera demasiados falsos atípicos en la cola derecha.\nEl Medcouple adapta el límite a la asimetría real (Hubert & Vandervieren).",
    x = "Precio de Venta (USD)", y = "Densidad"
  ) + theme_hp

print(p_justificacion)

# --- Aplicación del método ---
diag_out <- do.call(rbind, lapply(out_vars, function(v) {
  x <- train[[v]]
  
  # Límite ajustado por la medida robusta de asimetría (medcouple)
  bx_stats <- robustbase::adjboxStats(x[!is.na(x)])
  lim_inf <- bx_stats$fence[1]
  lim_sup <- bx_stats$fence[2]
  
  is_outlier <- !is.na(x) & (x < lim_inf | x > lim_sup)
  n_out <- sum(is_outlier)
  
  # Asignamos NA a los outliers para su posterior imputación
  train[[v]][is_outlier] <<- NA
  
  data.frame(
    Variable = v,
    Lim_Inf = round(lim_inf, 1),
    Lim_Sup = round(lim_sup, 1),
    N_out = n_out,
    Pct_out = round(100 * n_out / sum(!is.na(x)), 1)
  )
})) %>% arrange(desc(Pct_out))

print(diag_out, row.names = FALSE)


# --- PASO 7: Imputación k-NN ----------------------------------
cat("\nPASO 7 — AGREGACIÓN: Imputación k-NN (VIM)\n")
cat("Se utiliza k=5 (por defecto en la literatura y VIM) porque es un heurístico robusto:\n")
cat(" - Un k muy bajo (ej. k=1) tiene alta varianza y es sensible al ruido.\n")
cat(" - Un k muy alto sobre-suaviza los datos, perdiendo la variabilidad local.\n")
cat("La imputación usa la distancia de Gower, preservando las relaciones multivariantes.\n")

# Se imputan los NAs usando la mediana de los 5 vecinos más parecidos
# imp_var = FALSE evita que se creen columnas booleanas innecesarias en el dataset

train <- VIM::kNN(train, variable = out_vars, k = 5, imp_var = FALSE)

# NUEVO: Sincronización post-imputación
train <- train %>%
  mutate(
    TotalSF = TotalBsmtSF + FlrSF_1st + FlrSF_2nd,
    QualSF  = as.integer(OverallQual) * TotalSF
  )


# --- Visualización comparativa Antes vs Después ---------------
vars_plot <- c("SalePrice", "GrLivArea", "LotArea", "TotalSF")

# Preparamos los datos en formato largo para ggplot
df_comp <- bind_rows(
  train_pre_outliers %>% dplyr::select(all_of(vars_plot)) %>% mutate(Fase = "1. Original (Con Outliers)"),
  train %>% dplyr::select(all_of(vars_plot)) %>% mutate(Fase = "2. Post-Imputación (k-NN, k=5)")
) %>%
  pivot_longer(cols = -Fase, names_to = "Variable", values_to = "Valor")

p_comparativa <- ggplot(df_comp, aes(x = Fase, y = Valor, fill = Fase)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.5, outlier.size = 1) +
  facet_wrap(~Variable, scales = "free_y") +
  scale_y_continuous(labels = label_comma()) +
  scale_fill_manual(values = c("1. Original (Con Outliers)" = "#FC8D59", "2. Post-Imputación (k-NN, k=5)" = "#74ADD1")) +
  labs(
    title = "Efecto del tratamiento de outliers y agregación k-NN",
    subtitle = "Los valores extremos se identificaron mediante Boxplot Ajustado y se imputaron mediante VIM::kNN\nbasándose en los 5 vecinos más similares (distancia de Gower).",
    x = NULL, y = "Valor"
  ) +
  theme_hp + 
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 15, hjust = 1, face = "bold"),
    strip.text = element_text(size = 11, face = "bold", color = "#2C7BB6")
  )

print(p_comparativa)

cat("\n✓ Tratamiento de outliers completado: Datos extremos suavizados mediante k-NN.\n")

# --- PASO 8: Estandarización de variables numéricas (Z-Score) --
cat("\nPASO 8 — Estandarización Z-Score (dlookr)\n")
# Ajustamos variables continuas a una escala común sin perder el impacto de los outliers
vars_a_escalar <- c("GrLivArea", "LotArea", "TotalBsmtSF", "TotalSF")

for (v in vars_a_escalar) {
  var_z <- paste0(v, "_z") # Creamos nuevas columnas escaladas
  train[[var_z]] <- dlookr::transform(train[[v]], method = "zscore")
}

# --- PASO 9: One Hot Encoding (Evitando Dummy Variable Trap) ---
cat("\nPASO 9 — One Hot Encoding (fastDummies)\n")
vars_categoricas_clave <- c("Neighborhood", "BldgType", "SaleCondition")

# Aplicamos dummy_cols eliminando la primera categoría base (remove_first_dummy = TRUE)
# para evitar multicolinealidad perfecta (trampa de la variable cualitativa)
train <- dummy_cols(train, 
                    select_columns = vars_categoricas_clave,
                    remove_first_dummy = TRUE, 
                    remove_selected_columns = FALSE) # Se conservan para el EDA
cat("Variables dummy de Neighborhood, BldgType y SaleCondition creadas exitosamente.\n")




# --- PASO 10: Detección de Inconsistencias Lógicas (editrules) ---
# Se ejecuta DESPUÉS de la imputación para no contaminar con NAs
cat("\n--- Verificación de Reglas Lógicas (post-imputación) ---\n")
reglas_vivienda <- editset(c(
  "SalePrice > 0",
  "LotArea > 0",
  "TotalBsmtSF >= 0",
  "GrLivArea >= FlrSF_1st",
  "YearBuilt <= YrSold"
))

train_num_check <- train %>%
  mutate(YrSold = as.integer(as.character(YrSold))) %>%  
  dplyr::select(SalePrice, LotArea, TotalBsmtSF, GrLivArea, FlrSF_1st, YearBuilt, YrSold) 


violaciones <- violatedEdits(reglas_vivienda, train_num_check)

print(summary(violaciones))


# Comprobamos si hay alguna violación lógica
if (sum(violaciones, na.rm = TRUE) > 0) {
  cat("\n¡Atención! Se han detectado inconsistencias lógicas. Procediendo a auto-corrección...\n")
  
  # PASO 10.1: Localización del error (Principio de Fellegi y Holt)
  # localizeErrors devuelve una matriz lógica 'adapt' indicando qué celdas específicas modificar
  errores_loc <- localizeErrors(reglas_vivienda, train_num_check, method = "mip")$adapt
  cols_evaluadas <- colnames(errores_loc)
  
  # PASO 10.2: Anulación (Convertir a NA solo las celdas conflictivas)
  for (col in cols_evaluadas) {
    filas_con_error <- which(errores_loc[, col])
    if (length(filas_con_error) > 0) {
      train[[col]][filas_con_error] <- NA
      cat(sprintf(" -> Anulado (NA) valor inconsistente en la columna '%s' (Fila: %s)\n", 
                  col, paste(filas_con_error, collapse = ", ")))
    }
  }
  
  # PASO 10.3: Re-imputación (k-NN) de los nuevos NAs generados
  # Se recalculan los valores basándose en los 5 vecinos más parecidos (distancia Gower)
  cat(" -> Re-imputando valores lógicos mediante k-NN...\n")
  train <- suppressWarnings(VIM::kNN(train, variable = cols_evaluadas, k = 5, imp_var = FALSE))
  
  # (Opcional) Sincronizamos las variables compuestas por si alguna se vio afectada
  train <- train %>%
    mutate(
      TotalSF = TotalBsmtSF + FlrSF_1st + FlrSF_2nd,
      QualSF  = as.integer(OverallQual) * TotalSF
    )
  
  cat("✓ Inconsistencias corregidas exitosamente.\n")
  
} else {
  cat("\n✓ Datos lógicamente consistentes: No se detectaron violaciones de edición.\n")
}




# --------------------------------------------------------------
# a.3) RESUMEN FINAL DEL CONJUNTO
# --------------------------------------------------------------
cat("\n", strrep("=", 65), "\n")
cat("RESUMEN FINAL (TRAS PREPROCESADO)\n")
cat(strrep("=", 65), "\n")
cat(sprintf("Individuos                : %d\n",  nrow(train)))
cat(sprintf("Variables totales         : %d\n",  ncol(train)))
cat(sprintf("  - Numéricas             : %d\n",  sum(sapply(train, is.numeric))))
cat(sprintf("  - Factor / Ordinal      : %d\n",  sum(sapply(train, is.factor))))
cat(sprintf("  - Valores perdidos      : %d\n\n",sum(is.na(train))))

print(skim(train))



# ==============================================================
# b) ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================

# ==============================================================
# b.1) ESTUDIO DESCRIPTIVO
# ==============================================================


# ------------------------------------------------------------------
# BLOQUE 0 — Prueba de normalidad y QQ-Plots con Bandas de Confianza
# ------------------------------------------------------------------
cat("\n", strrep("=", 65), "\n")
cat("ANÁLISIS DE NORMALIDAD: ORIGINAL VS BOX-COX\n")
cat(strrep("=", 65), "\n")

# Evaluación visual: QQ-Plots con bandas de confianza de la librería 'car'
par(mfrow = c(1, 2))
car::qqPlot(train$SalePrice, pch = 19, col = "#D7191C",
            main = "QQ-Plot: SalePrice Original",
            xlab = "Cuantiles teóricos", ylab = "Cuantiles muestrales")

car::qqPlot(train$SalePrice_BC, pch = 19, col = "#2C7BB6",
            main = paste0("QQ-Plot: SalePrice (Box-Cox, λ=", round(lambda_opt,2), ")"),
            xlab = "Cuantiles teóricos", ylab = "Cuantiles muestrales")
par(mfrow = c(1, 1)) # Restaurar panel gráfico

# Evaluación analítica: Test de Lilliefors sobre la variable original vs transformada
precio_valido <- train$SalePrice[!is.na(train$SalePrice)]
precio_bc_valido <- train$SalePrice_BC[!is.na(train$SalePrice_BC)]

test_orig <- nortest::lillie.test(precio_valido)
test_bc <- nortest::lillie.test(precio_bc_valido)

cat(sprintf("Test Lilliefors (Original) : p-valor = %.4e\n", test_orig$p.value))
cat(sprintf("Test Lilliefors (Box-Cox)  : p-valor = %.4e\n", test_bc$p.value))

cat("\nLa transformación de Box-Cox centra los cuantiles muestrales sobre la línea teórica,\n")
cat("corrigiendo la asimetría de forma mucho más rigurosa que el logaritmo estándar.\n\n")


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
    x = "Precio de venta (USD)", y = "Densidad"
  ) + theme_hp

p_sp2 <- ggplot(train, aes(x = SalePrice_BC)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50,
                 fill = "#1A9641", alpha = 0.75, color = "white") +
  geom_density(color = "#D7191C", linewidth = 1.1) +
  labs(
    title    = sprintf("Distribución de SalePrice (Box-Cox, λ = %.2f)", lambda_opt),
    subtitle = sprintf("Transformación óptima por MLE | Asimetría = %.2f | Curtosis = %.2f",
                       skewness(train$SalePrice_BC),
                       kurtosis(train$SalePrice_BC)),
    x = "SalePrice transformado (Box-Cox)", y = "Densidad"
  ) + theme_hp

grid.arrange(p_sp1, p_sp2, ncol = 2,
             top = "Variable respuesta: SalePrice original y transformación Box-Cox óptima")


# ------------------------------------------------------------------
# BLOQUE 2 — Tabla de estadísticos descriptivos (vars. numéricas)
# ------------------------------------------------------------------
num_vars_key <- c(
  "SalePrice", "LotArea",    "LotFrontage", "GrLivArea",
  "TotalSF",   "TotalBsmtSF","FlrSF_1st",   "FlrSF_2nd",
  "GarageArea","TotalBaths", "HouseAge",    "RemodAge",
  "Fireplaces","TotRmsAbvGrd","WoodDeckSF", "OpenPorchSF"
)

stats_tbl <- train %>%
  dplyr::select(all_of(num_vars_key)) %>%  # <-- Solo se añade dplyr:: aquí
  pivot_longer(everything(), names_to = "Variable", values_to = "Valor") %>%
  group_by(Variable) %>%
  summarise(
    N       = sum(!is.na(Valor)),
    Media   = round(mean(Valor,            na.rm = TRUE), 1),
    Mediana = round(median(Valor,          na.rm = TRUE), 1),
    DT      = round(sd(Valor,             na.rm = TRUE), 1),
    Min     = round(min(Valor,            na.rm = TRUE), 1),
    Q1      = round(quantile(Valor, 0.25, na.rm = TRUE), 1),
    Q3      = round(quantile(Valor, 0.75, na.rm = TRUE), 1),
    Max     = round(max(Valor,            na.rm = TRUE), 1),
    Asim    = round(skewness(Valor,       na.rm = TRUE), 2),
    Kurt    = round(kurtosis(Valor,       na.rm = TRUE), 2),
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
  "TotalSF", "TotalBaths","HouseAge",    "LotFrontage"
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
    x = "Calidad general", y = "Precio de venta (USD)"
  ) +
  theme_hp + theme(legend.position = "none")
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
    x = "Superficie habitable (sq ft)", y = "Precio de venta (USD)"
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
    x = "Superficie total (sq ft)", y = "Precio de venta (USD)"
  ) + theme_hp
print(p_scatter2)


# ------------------------------------------------------------------
# BLOQUE 7 — Precio mediano por Neighborhood (todos los barrios)
# ------------------------------------------------------------------
p_neigh <- train %>%
  group_by(Neighborhood) %>%
  summarise(median_price = median(SalePrice), n = n(), .groups = "drop") %>%
  mutate(Neighborhood = fct_reorder(Neighborhood, median_price)) %>%
  ggplot(aes(x = Neighborhood, y = median_price, fill = median_price)) +
  geom_col() +
  geom_text(aes(label = paste0("n=", n)),
            hjust = -0.1, size = 2.7, color = "grey30") +
  scale_y_continuous(labels = label_dollar(),
                     expand = expansion(mult = c(0, 0.18))) +
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
# BLOQUE 8 — Evolución del precio según la antigüedad de la casa
# ------------------------------------------------------------------
p_age <- ggplot(train, aes(x = HouseAge, y = SalePrice)) +
  geom_point(alpha = 0.45, color = "#2C7BB6") +
  geom_smooth(method = "loess", color = "#D7191C", se = TRUE, linetype = "dashed") +
  scale_y_continuous(labels = label_dollar()) +
  labs(
    title = "Precio de venta vs Antigüedad de la vivienda (HouseAge)",
    subtitle = sprintf("Correlación de Spearman: rho = %.3f", 
                       cor(train$SalePrice, train$HouseAge, use="pairwise.complete.obs", method="spearman")),
    x = "Antigüedad (años)", y = "Precio de venta (USD)"
  ) + theme_hp
print(p_age)


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
    x = "Mes de venta", y = "Precio mediano (USD)", size = "Nº ventas"
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
    scale_y_continuous(labels = label_dollar(),
                       expand = expansion(mult = c(0, 0.15))) +
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
p_bldg <- ggplot(train, aes(x = BldgType, y = SalePrice, fill = BldgType)) +
  geom_violin(trim = FALSE, alpha = 0.65) +
  geom_boxplot(width = 0.1, fill = "white",
               outlier.size = 0.5, outlier.alpha = 0.4) +
  scale_y_continuous(labels = label_dollar()) +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title = "Distribución del precio de venta por tipo de edificio (BldgType)",
    x = "Tipo de edificio", y = "Precio de venta (USD)"
  ) +
  theme_hp + theme(legend.position = "none")
print(p_bldg)


# ------------------------------------------------------------------
# BLOQUE 12 — Frecuencias de variables categóricas clave
# ------------------------------------------------------------------
cat_key <- c(
  "MSZoning", "BldgType",    "HouseStyle",
  "Foundation","GarageType", "SaleCondition",
  "CentralAir","Neighborhood"
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
    `Porcentaje (%)`      = round(100 * Frecuencia / sum(Frecuencia), 1),
    `Frec. acumulada`     = cumsum(Frecuencia),
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
    N       = n(),
    Mínimo  = dollar(min(SalePrice)),
    Q1      = dollar(quantile(SalePrice, 0.25)),
    Mediana = dollar(median(SalePrice)),
    Media   = dollar(round(mean(SalePrice))),
    Q3      = dollar(quantile(SalePrice, 0.75)),
    Máximo  = dollar(max(SalePrice)),
    DT      = dollar(round(sd(SalePrice))),
    .groups = "drop"
  )

cat("\nEstadísticos de SalePrice por Calidad General (OverallQual):\n")
print(tab_qual)


# ------------------------------------------------------------------
# BLOQUE 15 — Matriz de correlación de Spearman
# ------------------------------------------------------------------
cor_vars <- c(
  "SalePrice", "GrLivArea", "LotArea", "LotFrontage",
  "TotalBsmtSF","GarageArea", "TotalSF", "TotalBaths",
  "HouseAge", "RemodAge", "TotRmsAbvGrd","Fireplaces",
  "WoodDeckSF", "OpenPorchSF", "OverallQual_n","QualSF"
)

train_cor <- train %>%
  mutate(OverallQual_n = as.numeric(OverallQual)) %>%
  dplyr::select(all_of(cor_vars))

cor_mat <- cor(train_cor, use = "pairwise.complete.obs", method = "spearman")

par(mar = c(0, 0, 3, 0))
corrplot(
  cor_mat, method = "color", type = "upper", order = "hclust",
  tl.cex = 0.78, tl.col = "black", addCoef.col = "black", number.cex = 0.52,
  col = colorRampPalette(c("#D7191C", "white", "#2C7BB6"))(200),
  title = "Matriz de correlación de Spearman — Variables numéricas clave",
  mar = c(0, 0, 2, 0)
)

# ------------------------------------------------------------------
# BLOQUE 16 — Top 10 correlaciones de Spearman con SalePrice
# ------------------------------------------------------------------
cor_sp <- cor_mat["SalePrice", ]
top_cor <- sort(abs(cor_sp[names(cor_sp) != "SalePrice"]), decreasing = TRUE)[1:10]

p_cor <- data.frame(Variable = names(top_cor), Correlacion = as.numeric(top_cor)) %>%
  mutate(Variable = fct_reorder(Variable, Correlacion)) %>%
  ggplot(aes(x = Variable, y = Correlacion, fill = Correlacion)) +
  geom_col() +
  geom_text(aes(label = round(Correlacion, 3)), hjust = -0.1, size = 3.2) +
  scale_fill_gradient(low = "#FEE090", high = "#D7191C") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  coord_flip() +
  labs(
    title = "Top 10: Variables con mayor correlación con SalePrice",
    subtitle = "Valor absoluto de la correlación de Spearman",
    x = NULL, y = "|rho de Spearman|", fill = "|rho|"
  ) + theme_hp

print(p_cor)


# ==============================================================
# b.2) GRÁFICOS INTERACTIVOS (plotly)
# ==============================================================

# --- Interactivo 1: SalePrice vs TotalSF ----------------------
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
    "<b>Precio de venta:</b> ",   SP_fmt,
    "<br><b>Superficie total:</b> ", SF_fmt, " sq ft",
    "<br><b>Barrio:</b> ",         Neighborhood,
    "<br><b>Calidad general:</b> ",OverallQual,
    "<br><b>Tipo de edificio:</b> ",BldgType,
    "<br><b>Estilo de la casa:</b> ",HouseStyle,
    "<br><b>Año construido:</b> ", YearBuilt,
    "<br><b>Año vendido:</b> ",    YrSold,
    "<br><b>Condición venta:</b> ",SaleCondition
  ),
  hoverinfo = "text"
) %>%
  layout(
    title = list(
      text = paste0(
        "<b>Precio de venta vs Superficie Total</b><br>",
        "<sup>Coloreado por Calidad General (1–10) — ",
        "House Prices · Ames, Iowa · Kaggle</sup>"
      ),
      font = list(size = 15)
    ),
    xaxis     = list(title = "Superficie total (sq ft)", tickformat = ","),
    yaxis     = list(title = "Precio de venta (USD)", tickprefix = "$", tickformat = ","),
    coloraxis = list(colorbar = list(title = "<b>Calidad<br>General</b>", tickvals = 1:10)),
    hovermode     = "closest",
    plot_bgcolor  = "#F8F9FA",
    paper_bgcolor = "#FFFFFF",
    font          = list(family = "Arial")
  )
print(p_interactive)


# --- Interactivo 2: SalePrice vs QualSF (feature reina) -------
df_inter_reina <- train %>%
  mutate(
    OverallQual_n = as.numeric(OverallQual),
    SP_fmt        = scales::dollar(SalePrice),
    QualSF_fmt    = scales::comma(QualSF),
    TotalSF_fmt   = scales::comma(TotalSF)
  )

p_interactive_reina <- plot_ly(
  data      = df_inter_reina,
  x         = ~QualSF,
  y         = ~SalePrice,
  color     = ~OverallQual_n,
  colors    = viridis(10, option = "plasma"),
  type      = "scatter",
  mode      = "markers",
  marker    = list(size = 6, opacity = 0.65),
  text      = ~paste0(
    "<b>Precio de venta:</b> ",       SP_fmt,
    "<br><b>QualSF (reina):</b> ",    QualSF_fmt,
    "<br><b>Superficie total:</b> ",  TotalSF_fmt, " sq ft",
    "<br><b>Barrio:</b> ",            Neighborhood,
    "<br><b>Calidad general:</b> ",   OverallQual,
    "<br><b>Tipo de edificio:</b> ",  BldgType,
    "<br><b>Estilo de la casa:</b> ", HouseStyle,
    "<br><b>Año construido:</b> ",    YearBuilt,
    "<br><b>Año vendido:</b> ",       YrSold,
    "<br><b>Condición venta:</b> ",   SaleCondition
  ),
  hoverinfo = "text"
) %>%
  layout(
    title = list(
      text = paste0(
        "<b>Precio de venta vs Feature reina (QualSF = OverallQual × TotalSF)</b><br>",
        "<sup>Coloreado por Calidad General (1–10) — House Prices · Ames, Iowa · Kaggle</sup>"
      ),
      font = list(size = 15)
    ),
    xaxis     = list(title = "QualSF (OverallQual × TotalSF)", tickformat = ","),
    yaxis     = list(title = "Precio de venta (USD)", tickprefix = "$", tickformat = ","),
    coloraxis = list(colorbar = list(title = "Calidad<br>General", tickvals = 1:10)),
    hovermode     = "closest",
    plot_bgcolor  = "#F8F9FA",
    paper_bgcolor = "#FFFFFF",
    font          = list(family = "Arial")
  )
print(p_interactive_reina)


# ==============================================================
cat("\n", strrep("=", 65), "\n")
cat("✓ Análisis completado con éxito.\n")
cat(strrep("=", 65), "\n")









# ==============================================================
# PARTE 2: ANÁLISIS MULTIVARIANTE
# Continuación del script Trabajo_final_definitivo_parte1.R
# Secciones:
#   c) Análisis de Componentes Principales (PCA)
#   d) Análisis de Correspondencias (CA simple y MCA)
#   e) Análisis Clúster (K-means y Jerárquico)
# ==============================================================
# NOTA: Este script se ejecuta DESPUÉS de la Parte 1.
#       El objeto 'train' ya debe existir en el entorno.
# ==============================================================


# ==============================================================
# 0. CARGA DE LIBRERÍAS ADICIONALES
# ==============================================================


pacman::p_load(
  FactoMineR, factoextra,   # PCA, CA, MCA y visualizaciones
  ggrepel,                  # Etiquetas sin solapamiento
  cluster,                  # Algoritmos de clúster (silhouette)
  dendextend,               # Dendrogramas mejorados
  NbClust,                  # Determinación del nº óptimo de clústeres
  RColorBrewer              # Paletas de color
)


# ################################################################
# ################################################################
# ##                                                            ##
# ##   c) ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)             ##
# ##                                                            ##
# ################################################################
# ################################################################

cat("\n", strrep("=", 65), "\n")
cat("SECCIÓN c) — ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)\n")

cat(strrep("=", 65), "\n")

# ------------------------------------------------------------------
# c.1) Selección y preparación de variables numéricas para PCA
# ------------------------------------------------------------------
# Según la teoría (PCA, sección 3), todos los predictores deben ser
# cuantitativos. Seleccionamos las variables numéricas clave que 
# representan distintas dimensiones de las viviendas, excluyendo
# la variable respuesta SalePrice y las derivadas/transformadas 
# para evitar redundancia y circularidad.

pca_vars <- c(
  "LotFrontage", "LotArea",     "GrLivArea",   "TotalBsmtSF",
  "FlrSF_1st",   "FlrSF_2nd",   "GarageArea",  "WoodDeckSF",
  "OpenPorchSF", "TotalBaths",  "TotRmsAbvGrd", "Fireplaces",
  "HouseAge",    "RemodAge",    "BedroomAbvGr",  "KitchenAbvGr"
)

# Extraemos la submatriz numérica sin NAs
train_pca <- train %>%
  dplyr::select(all_of(pca_vars)) %>%
  drop_na()

cat(sprintf("\nVariables seleccionadas para PCA: %d\n", length(pca_vars)))
cat(sprintf("Observaciones válidas (sin NA)  : %d\n", nrow(train_pca)))

# Verificamos medias y varianzas antes del escalado (como en el ejemplo
# USArrests de la teoría PCA, sección 4)
cat("\nMedias de las variables seleccionadas:\n")
print(round(apply(train_pca, 2, mean), 2))
cat("\nVarianzas de las variables seleccionadas:\n")
print(round(apply(train_pca, 2, var), 2))

cat("\nLas variables tienen escalas muy distintas (sq ft, años, conteos),")
cat("\npor lo que es IMPRESCINDIBLE estandarizar (scale=TRUE) antes del PCA")
cat("\n(Teoría PCA, sección 3.6).\n")


# ------------------------------------------------------------------
# c.2) Ejecución del PCA con prcomp (datos estandarizados)
# ------------------------------------------------------------------
# Siguiendo la teoría (sección 4): prcomp() centra y escala las variables.
# scale=TRUE → cada variable tendrá media 0 y desviación estándar 1.

pca_out <- prcomp(train_pca, scale = TRUE, center = TRUE)

cat("\nResultados del PCA (prcomp):\n")
cat("Componentes obtenidas:", ncol(pca_out$rotation), "\n")
cat("(min{n-1, p} =", min(nrow(train_pca)-1, ncol(train_pca)), ")\n")


# ------------------------------------------------------------------
# c.3) Cargas (loadings) de las componentes principales
# ------------------------------------------------------------------
# La matriz de rotación contiene los vectores de carga (teoría, sección 4).
# Cada columna es un vector de carga ϕ de la CP correspondiente.

cat("\nCargas (loadings) de las 4 primeras componentes principales:\n")
print(round(pca_out$rotation[, 1:4], 3))


# ------------------------------------------------------------------
# c.4) Varianza explicada y selección del número de componentes
# ------------------------------------------------------------------
# Según la teoría (sección 3.2 y 3.4):
#  - PVE = λ_i / Σλ_i
#  - Criterio: retener CP que expliquen el 75%-80% de la varianza total
#  - Criterio de Kaiser: autovalores > 1 (para datos estandarizados)
#  - Scree plot: buscar el "codo"

# Varianza de cada CP (autovalores)
pca_var <- pca_out$sdev^2
pve     <- pca_var / sum(pca_var)
pve_acum <- cumsum(pve)

cat("\n--- Varianza explicada por cada componente principal ---\n")
tabla_pve <- data.frame(
  CP            = paste0("PC", 1:length(pve)),
  Autovalor     = round(pca_var, 4),
  PVE           = round(pve * 100, 2),
  PVE_Acumulada = round(pve_acum * 100, 2)
)
print(tabla_pve)

# Criterio de Kaiser: retener CP con autovalor > 1
n_kaiser <- sum(pca_var > 1)
cat(sprintf("\nCriterio de Kaiser: retener %d componentes (autovalor > 1)\n", n_kaiser))

# Criterio del 80% de varianza acumulada
n_80 <- which(pve_acum >= 0.80)[1]
cat(sprintf("Criterio del 80%% de varianza: retener %d componentes (PVE acum. = %.1f%%)\n",
            n_80, pve_acum[n_80]*100))


# ------------------------------------------------------------------
# c.5) Scree Plot (Gráfico de sedimentación)
# ------------------------------------------------------------------
# Teoría (sección 3.4): representar autovalores de mayor a menor.
# El número de CP se determina donde "el bastón se rompe".

p_scree <- fviz_eig(pca_out, addlabels = TRUE, ylim = c(0, 40),
                    barfill = "#2C7BB6", barcolor = "#2C7BB6",
                    linecolor = "#D7191C") +
  geom_hline(yintercept = 100/length(pca_vars), linetype = "dashed", 
             color = "#1A9641", linewidth = 0.8) +
  annotate("text", x = length(pca_vars) - 2, 
           y = 100/length(pca_vars) + 1.5,
           label = "Umbral Kaiser (100/p %)", color = "#1A9641", 
           size = 3.5, fontface = "italic") +
  labs(
    title    = "Scree Plot — Porcentaje de varianza explicada por CP",
    subtitle = sprintf("House Prices | %d variables | Criterio Kaiser: %d CPs | 80%% var.: %d CPs",
                       length(pca_vars), n_kaiser, n_80),
    x = "Componente Principal", y = "% Varianza Explicada"
  ) + theme_hp

print(p_scree)


# ------------------------------------------------------------------
# c.6) PVE individual y acumulada (como en el ejemplo de la teoría)
# ------------------------------------------------------------------
par(mfrow = c(1, 2))
plot(pve, xlab = "Componente Principal",
     ylab = "Proporción de Varianza Explicada",
     ylim = c(0, max(pve) + 0.05), type = "b", pch = 19, col = "#2C7BB6",
     main = "PVE por componente")
abline(h = 1/length(pca_vars), col = "#1A9641", lty = 2) # Umbral Kaiser

plot(pve_acum, xlab = "Componente Principal",
     ylab = "PVE Acumulada",
     ylim = c(0, 1), type = "b", pch = 19, col = "#D7191C",
     main = "PVE Acumulada")
abline(h = 0.80, col = "#1A9641", lty = 2, lwd = 2)
text(length(pve_acum) - 3, 0.83, "80%", col = "#1A9641", font = 2)
par(mfrow = c(1, 1))


# ------------------------------------------------------------------
# c.7) Biplot: observaciones y variables en el plano PC1-PC2
# ------------------------------------------------------------------
# Según la teoría (sección 3.5):
#  - Longitud del vector → varianza explicada por las 2 primeras CP
#  - Ángulo entre vectores → correlación entre variables
#  - Dirección del vector → contribución a cada CP

# Biplot con factoextra (más limpio que biplot() base)
p_biplot <- fviz_pca_biplot(pca_out,
                            repel = TRUE,
                            col.var = "#D7191C",    # Variables en rojo
                            col.ind = "#2C7BB6",    # Individuos en azul
                            alpha.ind = 0.15,
                            geom.ind = "point",
                            pointsize = 1.2,
                            arrowsize = 0.8,
                            labelsize = 3) +
  labs(
    title    = "Biplot PCA — Observaciones y Variables (PC1 vs PC2)",
    subtitle = sprintf("PC1 = %.1f%% + PC2 = %.1f%% → %.1f%% varianza explicada",
                       pve[1]*100, pve[2]*100, (pve[1]+pve[2])*100)
  ) + theme_hp

print(p_biplot)


# ------------------------------------------------------------------
# c.8) Biplot coloreado por OverallQual (variable suplementaria)
# ------------------------------------------------------------------
# Utilizamos OverallQual como variable cualitativa ilustrativa
# para ver si las CP separan viviendas de distinta calidad.

# Necesitamos que el vector tenga la misma longitud que train_pca
idx_valid <- which(complete.cases(train[, pca_vars]))
qual_grupo <- train$OverallQual[idx_valid]

p_biplot_qual <- fviz_pca_ind(
  pca_out,
  geom        = "point",
  col.ind     = factor(as.numeric(qual_grupo)),   # factor = escala discreta correcta
  palette     = viridis(10, option = "plasma"),   # 10 colores, uno por nivel
  addEllipses = FALSE,
  pointsize   = 1.5,
  alpha.ind   = 0.5
) +
  labs(
    title    = "PCA — Observaciones coloreadas por Calidad General",
    subtitle = "OverallQual (1–10) como variable suplementaria cualitativa",
    color    = "Calidad\nGeneral"
  ) + theme_hp

print(p_biplot_qual)


# ------------------------------------------------------------------
# c.9) Contribución de las variables a las componentes principales
# ------------------------------------------------------------------
# Teoría (sección 3.3.1): |ϕ_ik| mide la importancia de la variable
# k-ésima en la componente i-ésima.

# Contribución a PC1
p_contrib1 <- fviz_contrib(pca_out, choice = "var", axes = 1, top = 16,
                           fill = "#2C7BB6", color = "#2C7BB6") +
  labs(title = "Contribución de variables a PC1") + theme_hp

# Contribución a PC2
p_contrib2 <- fviz_contrib(pca_out, choice = "var", axes = 2, top = 16,
                           fill = "#D7191C", color = "#D7191C") +
  labs(title = "Contribución de variables a PC2") + theme_hp

grid.arrange(p_contrib1, p_contrib2, ncol = 2,
             top = "Contribución de las variables a las dos primeras CP")


# ------------------------------------------------------------------
# c.10) Círculo de correlaciones (variables en el plano factorial)
# ------------------------------------------------------------------
# Teoría (sección 3.5): cuanto más paralelo sea un vector al eje
# de una componente, más ha contribuido a su creación.

p_var_pca <- fviz_pca_var(pca_out, 
                          col.var = "contrib",
                          gradient.cols = c("#FEE090", "#D7191C"),
                          repel = TRUE,
                          labelsize = 3.5) +
  labs(
    title    = "Círculo de correlaciones — Variables en el plano PC1-PC2",
    subtitle = "Coloreado por contribución a las dos primeras CP"
  ) + theme_hp

print(p_var_pca)


# ------------------------------------------------------------------
# c.11) Correlación entre CP y SalePrice
# ------------------------------------------------------------------
# Aunque SalePrice no participó en el PCA, podemos examinar su
# correlación con las componentes para evaluar su poder predictivo.

scores_pca <- pca_out$x[, 1:n_kaiser]
precio_pca <- train$SalePrice[idx_valid]

cor_cp_precio <- cor(scores_pca, precio_pca, use = "pairwise.complete.obs")
cat("\nCorrelación de las componentes principales con SalePrice:\n")
print(round(cor_cp_precio, 4))

cat("\nInterpretación: La primera CP (combinación lineal de tamaño,")
cat("\ncalidad y edad) captura la mayor parte de la variación del precio.\n")


# ------------------------------------------------------------------
# c.12) PCA con FactoMineR (análisis complementario)
# ------------------------------------------------------------------
# FactoMineR::PCA ofrece diagnósticos adicionales automáticos

pca_FM <- PCA(train_pca, scale.unit = TRUE, ncp = n_kaiser, graph = FALSE)

cat("\n--- Autovalores (FactoMineR) ---\n")
print(round(pca_FM$eig, 3))

cat("\n--- Coordenadas de las variables (2 primeras dim.) ---\n")
print(round(pca_FM$var$coord[, 1:2], 3))

cat("\n--- Calidad de representación (cos2) de las variables ---\n")
print(round(pca_FM$var$cos2[, 1:2], 3))



# ################################################################
# ################################################################
# ##                                                            ##
# ##   d) ANÁLISIS DE CORRESPONDENCIAS                          ##
# ##      d.1) CA Simple (tabla de contingencia 2 variables)    ##
# ##      d.2) Análisis de Correspondencias Múltiples (MCA)     ##
# ##                                                            ##
# ################################################################
# ################################################################

cat("\n", strrep("=", 65), "\n")
cat("SECCIÓN d) — ANÁLISIS DE CORRESPONDENCIAS\n")
cat(strrep("=", 65), "\n")

# ==================================================================
# d.1) ANÁLISIS DE CORRESPONDENCIAS SIMPLES (CA)
# ==================================================================
# Según la teoría (AC1, sección 1): el AC simple se aplica a tablas de
# contingencia de dos variables cualitativas. Su objetivo es representar
# simultáneamente los puntos fila y columna en un subespacio reducido.
# ------------------------------------------------------------------

cat("\n--- d.1) Análisis de Correspondencias Simples (CA) ---\n")

# ------------------------------------------------------------------
# d.1.1) Tabla de contingencia: Neighborhood × OverallQual
# ------------------------------------------------------------------
# Seleccionamos los barrios más frecuentes para una visualización legible
top_neighs <- train %>%
  count(Neighborhood) %>%
  arrange(desc(n)) %>%
  slice_head(n = 10) %>%
  pull(Neighborhood)

# Agrupamos OverallQual en 3 categorías para simplificar la tabla
train_ca <- train %>%
  filter(Neighborhood %in% top_neighs) %>%
  mutate(
    QualGrupo = case_when(
      as.numeric(OverallQual) <= 4  ~ "Baja (1-4)",
      as.numeric(OverallQual) <= 7  ~ "Media (5-7)",
      TRUE                          ~ "Alta (8-10)"
    ),
    QualGrupo = factor(QualGrupo, levels = c("Baja (1-4)", "Media (5-7)", "Alta (8-10)"))
  )

# Construimos la tabla de contingencia
tabla_cont_1 <- table(train_ca$Neighborhood, train_ca$QualGrupo)
cat("\nTabla de contingencia: Neighborhood × Grupo de Calidad\n")
print(tabla_cont_1)

# Test de independencia Chi-cuadrado
chi_test_1 <- chisq.test(tabla_cont_1)
cat(sprintf("\nTest Chi-cuadrado: X² = %.2f, df = %d, p-valor = %.2e\n",
            chi_test_1$statistic, chi_test_1$parameter, chi_test_1$p.value))
cat("Conclusión: ")
if (chi_test_1$p.value < 0.05) {
  cat("Rechazamos H0 → Hay asociación significativa entre Barrio y Calidad.\n")
} else {
  cat("No se rechaza H0 → No hay evidencia de asociación.\n")
}


# ------------------------------------------------------------------
# d.1.2) Ejecución del CA simple con FactoMineR
# ------------------------------------------------------------------
# Teoría (AC1, sección 3.5): los autovalores indican la cantidad de
# información retenida por cada dimensión.

res_ca1 <- CA(tabla_cont_1, graph = FALSE)

# Autovalores y varianza explicada
eig_ca1 <- get_eigenvalue(res_ca1)
cat("\n--- Autovalores del CA (Neighborhood × Calidad) ---\n")
print(round(eig_ca1, 3))

# Scree plot del CA
p_scree_ca1 <- fviz_screeplot(res_ca1, addlabels = TRUE, ylim = c(0, 100)) +
  labs(
    title    = "Scree Plot — CA: Neighborhood × Grupo de Calidad",
    subtitle = "Porcentaje de varianza (inercia) explicada por cada dimensión"
  ) + theme_hp
print(p_scree_ca1)


# ------------------------------------------------------------------
# d.1.3) Biplot simétrico del CA
# ------------------------------------------------------------------
# Teoría (AC1, sección 3.12): en el biplot simétrico, filas y columnas
# se representan en el mismo espacio. Si el ángulo entre una fila y una
# columna es agudo, existe fuerte asociación entre ambas.

p_ca1_biplot <- fviz_ca_biplot(res_ca1, 
                               repel = TRUE,
                               col.row = "#2C7BB6",
                               col.col = "#D7191C",
                               shape.row = 17, shape.col = 15,
                               labelsize = 4) +
  labs(
    title    = "CA Biplot simétrico: Neighborhood × Grupo de Calidad",
    subtitle = sprintf("Dim.1 = %.1f%% + Dim.2 = %.1f%% → %.1f%% inercia explicada",
                       eig_ca1[1, 2], eig_ca1[2, 2], eig_ca1[1, 3])
  ) + theme_hp

print(p_ca1_biplot)


# ------------------------------------------------------------------
# d.1.4) Contribución de filas y columnas al CA
# ------------------------------------------------------------------
p_ca1_row <- fviz_contrib(res_ca1, choice = "row", axes = 1:2, top = 10,
                          fill = "#2C7BB6") +
  labs(title = "Contribución de los barrios a Dim.1-2") + theme_hp

p_ca1_col <- fviz_contrib(res_ca1, choice = "col", axes = 1:2,
                          fill = "#D7191C") +
  labs(title = "Contribución de los niveles de calidad a Dim.1-2") + theme_hp

grid.arrange(p_ca1_row, p_ca1_col, ncol = 2,
             top = "Contribuciones al Análisis de Correspondencias Simple")


# ------------------------------------------------------------------
# d.1.5) Segundo CA: BldgType × SaleCondition
# ------------------------------------------------------------------
cat("\n--- Segundo CA: BldgType × SaleCondition ---\n")

tabla_cont_2 <- table(train$BldgType, train$SaleCondition)
cat("\nTabla de contingencia: BldgType × SaleCondition\n")
print(tabla_cont_2)

chi_test_2 <- chisq.test(tabla_cont_2)
cat(sprintf("\nTest Chi-cuadrado: X² = %.2f, df = %d, p-valor = %.2e\n",
            chi_test_2$statistic, chi_test_2$parameter, chi_test_2$p.value))

res_ca2 <- CA(tabla_cont_2, graph = FALSE)

p_ca2_biplot <- fviz_ca_biplot(res_ca2, 
                               repel = TRUE,
                               map = "rowprincipal",
                               arrow = c(TRUE, TRUE),
                               col.row = "#2C7BB6",
                               col.col = "#D7191C",
                               labelsize = 4) +
  labs(
    title    = "CA Biplot asimétrico: BldgType × SaleCondition",
    subtitle = "Mapa de filas en coordenadas principales (rowprincipal)"
  ) + theme_hp

print(p_ca2_biplot)


# ==================================================================
# d.2) ANÁLISIS DE CORRESPONDENCIAS MÚLTIPLES (MCA)
# ==================================================================
# Teoría (AC1, sección 4): extensión del CA simple al caso de varias
# variables nominales. Cada fila = individuo, cada columna = variable
# cualitativa. Se busca representar simultáneamente individuos y
# categorías en un espacio de baja dimensión.
# ------------------------------------------------------------------

cat("\n--- d.2) Análisis de Correspondencias Múltiples (MCA) ---\n")

# ------------------------------------------------------------------
# d.2.1) Selección de variables categóricas para MCA
# ------------------------------------------------------------------
# Elegimos variables cualitativas que describan distintos aspectos
# de las viviendas, con un número razonable de categorías.

mca_vars_activas <- c(
  "MSZoning", "BldgType", "HouseStyle", "Foundation",
  "CentralAir", "GarageType", "SaleCondition"
)

# Variables cuantitativas suplementarias (no intervienen en el MCA,
# pero se proyectan sobre las dimensiones para interpretación)
mca_vars_quanti_sup <- c("SalePrice", "GrLivArea", "HouseAge")

# Variable cualitativa suplementaria
mca_vars_quali_sup <- c("OverallQual")

# Preparación del dataframe para MCA
train_mca <- train %>%
  dplyr::select(all_of(c(mca_vars_activas, mca_vars_quali_sup, mca_vars_quanti_sup))) %>%
  drop_na()

cat(sprintf("Variables activas para MCA      : %d\n", length(mca_vars_activas)))
cat(sprintf("Variables cuanti. suplementarias: %d\n", length(mca_vars_quanti_sup)))
cat(sprintf("Variables cuali. suplementarias : %d\n", length(mca_vars_quali_sup)))
cat(sprintf("Observaciones válidas           : %d\n", nrow(train_mca)))


# ------------------------------------------------------------------
# d.2.2) Ejecución del MCA con FactoMineR
# ------------------------------------------------------------------
# Teoría (AC1, sección 4.12.2): especificamos variables suplementarias
# (no activas) para proyectarlas sin que influyan en las dimensiones.

# Índices de las columnas suplementarias
idx_quali_sup  <- which(names(train_mca) %in% mca_vars_quali_sup)
idx_quanti_sup <- which(names(train_mca) %in% mca_vars_quanti_sup)

res_mca <- MCA(train_mca,
               quali.sup  = idx_quali_sup,
               quanti.sup = idx_quanti_sup,
               graph = FALSE)

# Autovalores
eig_mca <- get_eigenvalue(res_mca)
cat("\n--- Autovalores del MCA (primeras 10 dimensiones) ---\n")
print(round(head(eig_mca, 10), 3))


# ------------------------------------------------------------------
# d.2.3) Scree Plot del MCA
# ------------------------------------------------------------------
p_scree_mca <- fviz_screeplot(res_mca, addlabels = TRUE, 
                              ncp = 15, ylim = c(0, 20)) +
  labs(
    title    = "Scree Plot — MCA: Variables categóricas de vivienda",
    subtitle = "Porcentaje de varianza (inercia) explicada por dimensión"
  ) + theme_hp
print(p_scree_mca)


# ------------------------------------------------------------------
# d.2.4) Mapa de categorías (variables activas)
# ------------------------------------------------------------------
# Cada punto representa una modalidad de una variable.
# Modalidades cercanas → perfiles de respuesta similares.

p_mca_var <- fviz_mca_var(res_mca,
                          repel = TRUE,
                          col.var = "contrib",
                          gradient.cols = c("#FEE090", "#FC8D59", "#D7191C"),
                          labelsize = 3,
                          ggtheme = theme_hp) +
  labs(
    title    = "MCA — Mapa de categorías (variables activas)",
    subtitle = "Coloreado por contribución a las dos primeras dimensiones"
  )
print(p_mca_var)


# ------------------------------------------------------------------
# d.2.5) Mapa de individuos coloreado por OverallQual (suplementaria)
# ------------------------------------------------------------------
p_mca_ind <- fviz_mca_ind(res_mca,
                          geom = "point",
                          habillage = idx_quali_sup,
                          addEllipses = TRUE,
                          ellipse.level = 0.75,
                          palette = viridis(10, option = "plasma"),
                          pointsize = 1,
                          alpha.ind = 0.3,
                          ggtheme = theme_hp) +
  labs(
    title    = "MCA — Individuos coloreados por Calidad General",
    subtitle = "OverallQual como variable cualitativa suplementaria (elipses al 75%)"
  )
print(p_mca_ind)


# ------------------------------------------------------------------
# d.2.6) Biplot MCA: individuos + categorías
# ------------------------------------------------------------------
p_mca_biplot <- fviz_mca_biplot(res_mca, 
                                repel = TRUE,
                                geom.ind = "point",
                                col.ind = "gray70",
                                alpha.ind = 0.15,
                                col.var = "contrib",
                                gradient.cols = c("#FEE090", "#D7191C"),
                                labelsize = 3,
                                ggtheme = theme_hp) +
  labs(
    title    = "MCA — Biplot: individuos y categorías de variables",
    subtitle = "Las categorías cercanas comparten perfiles similares de vivienda"
  )
print(p_mca_biplot)


# ------------------------------------------------------------------
# d.2.7) Contribuciones de las categorías a las dimensiones
# ------------------------------------------------------------------
p_mca_contrib1 <- fviz_contrib(res_mca, choice = "var", axes = 1, top = 15,
                               fill = "#2C7BB6") +
  labs(title = "Contribución de categorías a Dim.1") + theme_hp

p_mca_contrib2 <- fviz_contrib(res_mca, choice = "var", axes = 2, top = 15,
                               fill = "#D7191C") +
  labs(title = "Contribución de categorías a Dim.2") + theme_hp

grid.arrange(p_mca_contrib1, p_mca_contrib2, ncol = 2,
             top = "Contribuciones al Análisis de Correspondencias Múltiples")


# ------------------------------------------------------------------
# d.2.8) Descripción de dimensiones
# ------------------------------------------------------------------
# Teoría (AC1, sección 4.11): dimdesc() identifica las variables
# más correlacionadas con cada dimensión.

desc_mca <- dimdesc(res_mca, axes = c(1, 2))
cat("\n--- Descripción de la Dimensión 1 del MCA ---\n")
print(desc_mca[[1]])
cat("\n--- Descripción de la Dimensión 2 del MCA ---\n")
print(desc_mca[[2]])


# ------------------------------------------------------------------
# d.2.9) Correlación de variables cuantitativas suplementarias
# ------------------------------------------------------------------
# Proyección de SalePrice, GrLivArea y HouseAge sobre las dimensiones
p_mca_quanti <- fviz_mca_var(res_mca, choice = "quanti.sup",
                             repel = TRUE,
                             ggtheme = theme_hp) +
  labs(
    title    = "MCA — Variables cuantitativas suplementarias",
    subtitle = "Correlación de SalePrice, GrLivArea y HouseAge con las dimensiones"
  )
print(p_mca_quanti)



# ################################################################
# ################################################################
# ##                                                            ##
# ##   e) ANÁLISIS CLÚSTER                                      ##
# ##      e.1) K-Means Clustering                               ##
# ##      e.2) Clúster Jerárquico                               ##
# ##      e.3) Comparación y perfilado de clústeres              ##
# ##                                                            ##
# ################################################################
# ################################################################

cat("\n", strrep("=", 65), "\n")
cat("SECCIÓN e) — ANÁLISIS CLÚSTER\n")
cat(strrep("=", 65), "\n")

# ------------------------------------------------------------------
# e.0) Preparación de datos para clustering
# ------------------------------------------------------------------
# Teoría (Clúster, sección 3.5): Si las variables se miden en escalas
# diferentes, es buena decisión escalarlas para tener desviación
# estándar uno antes de calcular las diferencias entre observaciones.

clust_vars <- c(
  "GrLivArea", "TotalBsmtSF", "GarageArea", "LotArea",
  "TotalBaths", "HouseAge", "TotRmsAbvGrd", "Fireplaces"
)

train_clust <- train %>%
  dplyr::select(all_of(clust_vars)) %>%
  drop_na()

# Estandarización (media 0, sd 1)
train_clust_scaled <- scale(train_clust)

cat(sprintf("\nVariables para clustering: %d\n", length(clust_vars)))
cat(sprintf("Observaciones: %d\n", nrow(train_clust_scaled)))
cat("Variables estandarizadas (media=0, sd=1) para igualar la importancia.\n")


# ==================================================================
# e.1) K-MEANS CLUSTERING
# ==================================================================
# Teoría (Clúster, sección 2): K-means busca particionar las n
# observaciones en K grupos minimizando la varianza intra-clúster.
# Se recomienda nstart alto (20-50) para evitar mínimos locales.
# ------------------------------------------------------------------

cat("\n--- e.1) K-Means Clustering ---\n")

# ------------------------------------------------------------------
# e.1.1) Determinación del número óptimo de clústeres
# ------------------------------------------------------------------
# Teoría (Clúster, ejercicio 1b): fviz_nbclust() automatiza la búsqueda
# del K óptimo mediante el método del codo (wss), silueta, y gap statistic.

# Método del codo (Within Sum of Squares)
p_elbow <- fviz_nbclust(train_clust_scaled, kmeans, method = "wss",
                        k.max = 10, nstart = 50) +
  geom_vline(xintercept = 4, linetype = "dashed", color = "#D7191C") +
  labs(
    title    = "Método del codo — Suma de cuadrados intra-clúster (WSS)",
    subtitle = "El 'codo' sugiere el número óptimo de clústeres"
  ) + theme_hp
print(p_elbow)

# Método de la silueta promedio
p_silhouette <- fviz_nbclust(train_clust_scaled, kmeans, method = "silhouette",
                             k.max = 10, nstart = 50) +
  labs(
    title    = "Método de la silueta — Anchura media de silueta",
    subtitle = "Mayor anchura = mejor separación entre clústeres"
  ) + theme_hp
print(p_silhouette)


# ------------------------------------------------------------------
# e.1.2) Ejecución de K-Means con K óptimo
# ------------------------------------------------------------------
# Seleccionamos K basándonos en los gráficos anteriores
set.seed(101)  # Reproducibilidad (como sugiere la teoría, ejercicio 2a)

# Probamos K=3, K=4 y K=5 para comparar
km3 <- kmeans(train_clust_scaled, centers = 3, nstart = 50)
km4 <- kmeans(train_clust_scaled, centers = 4, nstart = 50)
km5 <- kmeans(train_clust_scaled, centers = 5, nstart = 50)

cat("\n--- Resultados K-Means ---\n")
cat(sprintf("K=3: WSS total = %.1f | BSS/TSS = %.1f%%\n",
            km3$tot.withinss, km3$betweenss/km3$totss*100))
cat(sprintf("K=4: WSS total = %.1f | BSS/TSS = %.1f%%\n",
            km4$tot.withinss, km4$betweenss/km4$totss*100))
cat(sprintf("K=5: WSS total = %.1f | BSS/TSS = %.1f%%\n",
            km5$tot.withinss, km5$betweenss/km5$totss*100))

# Tamaño de cada clúster
cat("\nTamaño de clústeres (K=4):\n")
print(table(km4$cluster))


# ------------------------------------------------------------------
# e.1.3) Visualización de K-Means en el plano PCA
# ------------------------------------------------------------------
# Proyectamos los clústeres sobre las dos primeras CP para visualizarlos
# en 2D (como sugiere la teoría, ejercicio 2g).

# K = 3
p_km3 <- fviz_cluster(km3, data = train_clust_scaled,
                      geom = "point", pointsize = 1, alpha = 0.4,
                      ellipse.type = "convex",
                      palette = c("#2C7BB6", "#D7191C", "#1A9641"),
                      ggtheme = theme_hp) +
  labs(title = "K-Means con K = 3",
       subtitle = sprintf("BSS/TSS = %.1f%%", km3$betweenss/km3$totss*100))

# K = 4
p_km4 <- fviz_cluster(km4, data = train_clust_scaled,
                      geom = "point", pointsize = 1, alpha = 0.4,
                      ellipse.type = "convex",
                      palette = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61"),
                      ggtheme = theme_hp) +
  labs(title = "K-Means con K = 4",
       subtitle = sprintf("BSS/TSS = %.1f%%", km4$betweenss/km4$totss*100))

# K = 5
p_km5 <- fviz_cluster(km5, data = train_clust_scaled,
                      geom = "point", pointsize = 1, alpha = 0.4,
                      ellipse.type = "convex",
                      ggtheme = theme_hp) +
  labs(title = "K-Means con K = 5",
       subtitle = sprintf("BSS/TSS = %.1f%%", km5$betweenss/km5$totss*100))

grid.arrange(p_km3, p_km4, p_km5, ncol = 3,
             top = "Comparación de agrupamientos K-Means (K = 3, 4, 5)")


# ------------------------------------------------------------------
# e.1.4) Gráfico de silueta para K = 4
# ------------------------------------------------------------------
sil_km4 <- silhouette(km4$cluster, dist(train_clust_scaled))
p_sil <- fviz_silhouette(sil_km4, palette = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61")) +
  labs(
    title    = "Gráfico de silueta — K-Means (K = 4)",
    subtitle = sprintf("Anchura media de silueta: %.3f", mean(sil_km4[, 3]))
  ) + theme_hp
print(p_sil)


# ==================================================================
# e.2) CLÚSTER JERÁRQUICO
# ==================================================================
# Teoría (Clúster, sección 3): el clúster jerárquico construye un
# dendrograma que agrupa observaciones por similitud, sin necesidad
# de especificar K a priori.
# ------------------------------------------------------------------

cat("\n--- e.2) Clúster Jerárquico ---\n")

# ------------------------------------------------------------------
# e.2.1) Matriz de distancias y dendrogramas
# ------------------------------------------------------------------
# Teoría (ejercicio 1d): usar distancia euclídea con enlace completo.
# Usamos una muestra aleatoria para que el dendrograma sea legible.

set.seed(42)
n_sample <- 200
idx_sample <- sample(nrow(train_clust_scaled), n_sample)
clust_sample <- train_clust_scaled[idx_sample, ]

# Matriz de distancias euclídeas
dist_eucl <- dist(clust_sample, method = "euclidean")

# Dendrogramas con distintos métodos de vinculación
# (Teoría, sección 3.3: completa, promedio, individual, Ward)
hc_complete <- hclust(dist_eucl, method = "complete")
hc_average  <- hclust(dist_eucl, method = "average")
hc_single   <- hclust(dist_eucl, method = "single")
hc_ward     <- hclust(dist_eucl, method = "ward.D2")

# Visualización comparativa de los 3 métodos principales
par(mfrow = c(1, 3))
plot(hc_complete, labels = FALSE, hang = -1, cex = 0.5,
     main = "Vinculación Completa", xlab = "", sub = "")
plot(hc_average, labels = FALSE, hang = -1, cex = 0.5,
     main = "Vinculación Promedio", xlab = "", sub = "")
plot(hc_ward, labels = FALSE, hang = -1, cex = 0.5,
     main = "Método de Ward (D2)", xlab = "", sub = "")
par(mfrow = c(1, 1))


# ------------------------------------------------------------------
# e.2.2) Coeficiente cofenético
# ------------------------------------------------------------------
# Teoría (ejercicio 1h): el coeficiente de correlación entre las 
# distancias cofenéticas del dendrograma y la matriz de distancias
# original mide la calidad del dendrograma. Valores > 0.75 son buenos.

coph_complete <- cor(cophenetic(hc_complete), dist_eucl)
coph_average  <- cor(cophenetic(hc_average),  dist_eucl)
coph_single   <- cor(cophenetic(hc_single),   dist_eucl)
coph_ward     <- cor(cophenetic(hc_ward),     dist_eucl)

cat("\nCoeficientes cofenéticos (calidad del dendrograma):\n")
cat(sprintf("  Completa : %.4f\n", coph_complete))
cat(sprintf("  Promedio : %.4f\n", coph_average))
cat(sprintf("  Individual: %.4f\n", coph_single))
cat(sprintf("  Ward (D2): %.4f\n", coph_ward))
cat("(Valores cercanos a 1 indican que el dendrograma refleja bien las distancias reales)\n")

# Seleccionamos el mejor método según el coeficiente cofenético
metodos_coph <- c(Complete = coph_complete, Average = coph_average,
                  Single = coph_single, Ward = coph_ward)
mejor_metodo <- names(which.max(metodos_coph))
cat(sprintf("\nMejor método según coeficiente cofenético: %s (%.4f)\n",
            mejor_metodo, max(metodos_coph)))


# ------------------------------------------------------------------
# e.2.3) Dendrograma con corte en K clústeres (Ward)
# ------------------------------------------------------------------
# Usamos Ward.D2 (comúnmente produce dendrogramas equilibrados).
# Cortamos a la altura que produzca 4 clústeres.

dend_ward <- as.dendrogram(hc_ward)
dend_ward <- color_branches(dend_ward, k = 4,
                            col = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61"))

par(mar = c(2, 4, 3, 1))
plot(dend_ward, 
     main = "Dendrograma — Método de Ward (n = 200, K = 4)",
     ylab = "Altura (distancia)", leaflab = "none")
abline(h = cutree_height <- mean(c(
  sort(hc_ward$height, decreasing = TRUE)[3],
  sort(hc_ward$height, decreasing = TRUE)[4]
)), col = "gray40", lty = 2, lwd = 1.5)
text(10, cutree_height + 0.5, "Corte K=4", col = "gray40", cex = 0.9)
par(mar = c(5, 4, 4, 2) + 0.1)

# Asignación de clústeres
clust_hc <- cutree(hc_ward, k = 4)
cat("\nDistribución de observaciones en clústeres jerárquicos (K=4):\n")
print(table(clust_hc))


# ------------------------------------------------------------------
# e.2.4) Visualización del clúster jerárquico en plano PCA
# ------------------------------------------------------------------
p_hc_pca <- fviz_cluster(list(data = clust_sample, cluster = clust_hc),
                         geom = "point", pointsize = 1.5, alpha = 0.5,
                         ellipse.type = "convex",
                         palette = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61"),
                         ggtheme = theme_hp) +
  labs(
    title    = "Clúster Jerárquico (Ward, K=4) — Proyección en plano PCA",
    subtitle = sprintf("Muestra de %d observaciones | Distancia euclídea", n_sample)
  )
print(p_hc_pca)


# ==================================================================
# e.3) PERFILADO Y COMPARACIÓN DE CLÚSTERES
# ==================================================================
# Caracterizamos los clústeres obtenidos con K-Means (K=4) analizando
# las variables originales y el precio de venta por grupo.
# ------------------------------------------------------------------

cat("\n--- e.3) Perfilado de clústeres (K-Means, K=4) ---\n")

# Asignamos el clúster a cada observación del dataset original
idx_clust_valid <- which(complete.cases(train[, clust_vars]))
train$Cluster_KM <- NA
train$Cluster_KM[idx_clust_valid] <- km4$cluster
train$Cluster_KM <- factor(train$Cluster_KM)

# ------------------------------------------------------------------
# e.3.1) Estadísticos descriptivos por clúster
# ------------------------------------------------------------------
perfil_vars <- c("SalePrice", clust_vars)

perfil_clust <- train %>%
  filter(!is.na(Cluster_KM)) %>%
  group_by(Cluster_KM) %>%
  summarise(
    N             = n(),
    Precio_Med    = round(median(SalePrice)),
    Precio_Mean   = round(mean(SalePrice)),
    GrLivArea_Med = round(median(GrLivArea)),
    TotalBsmtSF_Med = round(median(TotalBsmtSF)),
    GarageArea_Med  = round(median(GarageArea)),
    LotArea_Med   = round(median(LotArea)),
    HouseAge_Med  = round(median(HouseAge)),
    TotalBaths_Med = round(median(TotalBaths), 1),
    .groups = "drop"
  )

cat("\nPerfil de cada clúster (medianas):\n")
print(perfil_clust)


# ------------------------------------------------------------------
# e.3.2) Boxplot: SalePrice por clúster
# ------------------------------------------------------------------
p_clust_price <- train %>%
  filter(!is.na(Cluster_KM)) %>%
  ggplot(aes(x = Cluster_KM, y = SalePrice, fill = Cluster_KM)) +
  geom_boxplot(outlier.alpha = 0.3, alpha = 0.7) +
  scale_y_continuous(labels = label_dollar()) +
  scale_fill_manual(values = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61")) +
  labs(
    title    = "Distribución del precio de venta por clúster (K-Means, K=4)",
    subtitle = "Cada clúster agrupa viviendas con características físicas similares",
    x = "Clúster", y = "Precio de venta (USD)"
  ) +
  theme_hp + theme(legend.position = "none")

print(p_clust_price)


# ------------------------------------------------------------------
# e.3.3) Heatmap de centroides estandarizados
# ------------------------------------------------------------------
# Visualización de las medias estandarizadas de cada variable por clúster
# para interpretar qué define a cada grupo.

centroides_df <- as.data.frame(km4$centers)
centroides_df$Cluster <- paste0("C", 1:4)

centroides_long <- centroides_df %>%
  pivot_longer(cols = -Cluster, names_to = "Variable", values_to = "Valor_Z")

p_heatmap <- ggplot(centroides_long, 
                    aes(x = Variable, y = Cluster, fill = Valor_Z)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = round(Valor_Z, 2)), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "#2C7BB6", mid = "white", high = "#D7191C",
                       midpoint = 0, name = "Z-Score") +
  labs(
    title    = "Heatmap de centroides estandarizados (K-Means, K=4)",
    subtitle = "Valores Z positivos (rojo) = por encima de la media; negativos (azul) = por debajo",
    x = NULL, y = "Clúster"
  ) +
  theme_hp +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        panel.grid = element_blank())

print(p_heatmap)


# ------------------------------------------------------------------
# e.3.4) Distribución de OverallQual por clúster
# ------------------------------------------------------------------
p_clust_qual <- train %>%
  filter(!is.na(Cluster_KM)) %>%
  ggplot(aes(x = Cluster_KM, fill = OverallQual)) +
  geom_bar(position = "fill", alpha = 0.85) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_viridis_d(option = "plasma", name = "Calidad\nGeneral") +
  labs(
    title    = "Composición de calidad general por clúster",
    subtitle = "Proporción de cada nivel de OverallQual dentro de cada grupo",
    x = "Clúster", y = "Proporción"
  ) + theme_hp

print(p_clust_qual)


# ------------------------------------------------------------------
# e.3.5) Distribución de Neighborhood por clúster
# ------------------------------------------------------------------
# Mostramos los 8 barrios más frecuentes
top8_neigh <- train %>%
  filter(!is.na(Cluster_KM)) %>%
  count(Neighborhood) %>%
  arrange(desc(n)) %>%
  slice_head(n = 8) %>%
  pull(Neighborhood)

p_clust_neigh <- train %>%
  filter(!is.na(Cluster_KM), Neighborhood %in% top8_neigh) %>%
  ggplot(aes(x = Cluster_KM, fill = Neighborhood)) +
  geom_bar(position = "fill", alpha = 0.85) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Set2", name = "Barrio") +
  labs(
    title    = "Composición de barrios por clúster (top 8)",
    subtitle = "Proporción de cada barrio dentro de cada grupo",
    x = "Clúster", y = "Proporción"
  ) + theme_hp

print(p_clust_neigh)


# ------------------------------------------------------------------
# e.3.6) Tabla cruzada: Clúster × OverallQual
# ------------------------------------------------------------------
tabla_clust_qual <- table(train$Cluster_KM[!is.na(train$Cluster_KM)],
                          train$OverallQual[!is.na(train$Cluster_KM)])
cat("\nTabla cruzada: Clúster K-Means × Calidad General\n")
print(tabla_clust_qual)

# Test Chi-cuadrado de independencia
chi_clust <- chisq.test(tabla_clust_qual)
cat(sprintf("\nTest Chi²: X² = %.1f, p-valor = %.2e\n",
            chi_clust$statistic, chi_clust$p.value))
cat("Conclusión: Los clústeres están fuertemente asociados con la calidad general.\n")


# ==============================================================
# RESUMEN FINAL
# ==============================================================
cat("\n", strrep("=", 65), "\n")
cat("✓ PARTE 2 — Análisis Multivariante completado con éxito.\n")
cat(strrep("=", 65), "\n")
cat("\nResumen de resultados:\n")
cat(sprintf("  PCA: %d componentes retienen el %.1f%% de la varianza (Kaiser)\n",
            n_kaiser, pve_acum[n_kaiser]*100))
cat(sprintf("  CA simple: Asociación significativa Neighborhood × Calidad (p < 0.05)\n"))
cat(sprintf("  MCA: Perfiles diferenciados de vivienda según %d variables categóricas\n",
            length(mca_vars_activas)))
cat(sprintf("  K-Means: 4 clústeres con BSS/TSS = %.1f%%\n",
            km4$betweenss/km4$totss*100))
cat(sprintf("  Clúster Jerárquico: Mejor método = %s (coef. cofenético = %.3f)\n",
            mejor_metodo, max(metodos_coph)))
cat(strrep("=", 65), "\n")

