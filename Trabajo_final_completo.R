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
  robustbase, VIM, editrules, MASS, car, dlookr, fastDummies,
  fastcluster, mclust
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

set.seed(42)

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

# Ajustar la ruta si necesario
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
# ---> ELIMINACIÓN MANUAL DE OUTLIERS FAMOSOS
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

# Suponemos que para estas variables que sean na's significa que no tienen
# de tal característica, imputamos por 0 (por ejemplo, área del garage)
na_zero_num <- c(
  "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
  "TotalBsmtSF","BsmtFullBath", "BsmtHalfBath",
  "GarageCars", "GarageArea", "GarageYrBlt"
)

# Imputamos LotFrontage según el barrio (las casas tienen distribuciones
# similares al estar en el mismo barrio)
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

# Electrical tiene un solo na, imputamos por moda (variable categórica)
moda_elec <- names(which.max(table(train$Electrical)))
train$Electrical[is.na(train$Electrical)] <- moda_elec


# --- PASO 3: Feature Engineering ------------------------------
# Se extraen las nuevas variables ANTES de convertir YrSold a factor
# porque las operaciones aritméticas requieren valores numéricos.
yr_num <- as.integer(as.character(train$YrSold))

# Transformación de Box-Cox rigurosa (Sustituye al logaritmo manual logSalePrice)
# 1. Obtenemos el vector limpio sin NAs y estrictamente positivo
precio_limpio <- train$SalePrice[!is.na(train$SalePrice) & train$SalePrice > 0]

# 2. Buscamos el parámetro lambda óptimo maximizando la log-verosimilitud
b_cox <- boxcox(lm(precio_limpio ~ 1), plotit = FALSE)
lambda_opt <- b_cox$x[which.max(b_cox$y)]
cat(sprintf("\nLambda óptimo de Box-Cox para SalePrice: %.4f\n", lambda_opt))

# Creamos nuevas variables que nos pueden ayudar a hacer el análisis descriptivo
train <- train %>%
  mutate(
    SalePrice_BC  = (SalePrice^lambda_opt - 1) / lambda_opt,
    HouseAge      = yr_num - YearBuilt,
    RemodAge      = yr_num - YearRemodAdd,
    TotalSF       = TotalBsmtSF + FlrSF_1st + FlrSF_2nd,
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
# no una magnitud, por lo que la hacemos factor nominal
train$MSSubClass <- factor(train$MSSubClass)

# OverallQual / OverallCond: escala 1–10 con orden real → ordinal
train$OverallQual <- factor(train$OverallQual, levels = 1:10, ordered = TRUE)
train$OverallCond <- factor(train$OverallCond, levels = 1:10, ordered = TRUE)

# MoSold: número del mes → factor con etiquetas legibles
train$MoSold <- factor(train$MoSold, levels = 1:12, labels = month.abb)

# YrSold: año de venta → factor nominal
train$YrSold <- factor(train$YrSold)

# Variables de calidad con escala estandarizada Po-Ex (Poor-Excellent)
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

# Sincronización post-imputación
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
  cat(" -> Re-imputando valores lógicos mediante k-NN...\n")
  train <- suppressWarnings(VIM::kNN(train, variable = cols_evaluadas, k = 5, imp_var = FALSE))
  
  # Sincronizamos las variables compuestas por si alguna se vio afectada
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
cat("A pesar de que rechazamos normalidad en ambos casos, la diferencia del pvalor
    es muy grande (tras la transformación estamos mucho más cerca de conseguir
    que la distribución sea normal")


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
  dplyr::select(all_of(num_vars_key)) %>%
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
# PARTE 1.5: TABLAS DE CONTINGENCIA Y MEDIDAS DE ASOCIACIÓN
# ==============================================================
# Secciones:
#   c.1) Variables categóricas derivadas
#   c.2) Tabla 2×2: CentralAir × PriceCat (nominales)
#   c.3) Tabla N×M: Neighborhood × PriceCat (nominales)
#   c.4) Medidas de asociación nominales (C, Phi, V, Q de Yule)
#   c.5) Tabla ordinal: QualCat × AgeCat + medidas ordinales
#   c.6) Tabla ordinal: OverallQual × KitchenQual + D de Somers
#   c.7) Tabla tridimensional y CMH
#   c.8) Tabla mixta nominal × ordinal
#   c.9) Resumen global
# ==============================================================

cat("\n", strrep("=", 65), "\n")
cat("PARTE 1.5 — TABLAS DE CONTINGENCIA Y MEDIDAS DE ASOCIACIÓN\n")

cat(strrep("=", 65), "\n")

pacman::p_load(DescTools, vcd, rcompanion)


# ------------------------------------------------------------------
# FUNCIÓN AUXILIAR: test_independencia()
# ------------------------------------------------------------------
# Aplica automáticamente chi-cuadrado o Fisher según la condición
# del 80%: al menos el 80% de las frecuencias esperadas deben ser > 5.
# Si la condición NO se cumple → fisher.test() con simulación de p-valor
# (simulate.p.value = TRUE) para tablas grandes (> 2×2).
# Referencia: ADAT_medidas_de_asociacion.pdf, sección 2.1
# ------------------------------------------------------------------
test_independencia <- function(tabla, nombre = "") {
  cat(sprintf("\n--- Test de independencia: %s ---\n", nombre))
  
  # 1. Frecuencias esperadas bajo H0 de independencia
  esperadas <- suppressWarnings(chisq.test(tabla, correct = FALSE)$expected)
  pct_ok    <- round(100 * mean(esperadas >= 5), 1)
  
  cat(sprintf("  Celdas con frec. esperada ≥ 5: %.1f%%  (umbral mínimo: 80%%)\n", pct_ok))
  
  # 2. Decisión: chi-cuadrado vs Fisher
  if (pct_ok >= 80) {
    cat("  ✓ Condición cumplida → se aplica Chi-cuadrado de Pearson.\n")
    res <- chisq.test(tabla, correct = FALSE)
    cat(sprintf("  X² = %.2f,  gl = %d,  p-valor = %.2e\n",
                res$statistic, res$parameter, res$p.value))
  } else {
    cat("  ✗ Condición NO cumplida → se aplica Test Exacto de Fisher.\n")
    simular <- nrow(tabla) > 2 | ncol(tabla) > 2
    res <- fisher.test(tabla, simulate.p.value = simular, B = 10000)
    cat(sprintf("  p-valor = %.2e%s\n", res$p.value,
                ifelse(simular, "  (Monte Carlo, B = 10 000)", "")))
  }
  
  # 3. Conclusión
  cat("  Conclusión: ")
  if (res$p.value < 0.05) {
    cat("Rechazamos H0 → Hay asociación significativa entre las variables.\n")
  } else {
    cat("No se rechaza H0 → No hay evidencia de asociación significativa.\n")
  }
  
  invisible(res)
}


# ==============================================================
# c.1) CREACIÓN DE VARIABLES CATEGÓRICAS DERIVADAS
# ==============================================================

cat("\n--- c.1) Creación de variables categóricas para tablas de contingencia ---\n")

# Variable dicotómica de precio (como cat_wage en el PDF)
train$PriceCat <- factor(
  ifelse(train$SalePrice > median(train$SalePrice, na.rm = TRUE), "Alto", "Bajo"),
  levels = c("Bajo", "Alto")
)
cat(sprintf("Mediana de SalePrice: %s\n", scales::dollar(median(train$SalePrice, na.rm = TRUE))))
cat("Distribución de PriceCat:\n")
print(table(train$PriceCat))

# Variable de antigüedad categorizada (ordinal)
train$AgeCat <- cut(
  train$HouseAge,
  breaks = c(-Inf, 10, 25, 50, Inf),
  labels = c("Nueva (0-10)", "Reciente (11-25)", "Madura (26-50)", "Antigua (>50)"),
  ordered_result = TRUE
)

# Variable de superficie categorizada (ordinal)
train$SizeCat <- cut(
  train$GrLivArea,
  breaks = quantile(train$GrLivArea, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE),
  labels = c("Pequeña", "Mediana", "Grande"),
  include.lowest = TRUE,
  ordered_result = TRUE
)

# Variable de calidad agrupada (ordinal)
train$QualCat <- cut(
  as.numeric(train$OverallQual),
  breaks = c(0, 4, 6, 8, 10),
  labels = c("Baja (1-4)", "Media (5-6)", "Alta (7-8)", "Excelente (9-10)"),
  ordered_result = TRUE
)

cat("\nDistribución de AgeCat:\n");  print(table(train$AgeCat))
cat("\nDistribución de SizeCat:\n"); print(table(train$SizeCat))
cat("\nDistribución de QualCat:\n"); print(table(train$QualCat))


# ==============================================================
# c.2) TABLA DE CONTINGENCIA 2×2: PriceCat × CentralAir
# ==============================================================
# Referencia PDF sección 1.1

cat("\n", strrep("-", 55), "\n")
cat("--- c.2) Tabla de contingencia 2×2: CentralAir × PriceCat ---\n")
cat(strrep("-", 55), "\n")

con_2x2 <- table(train$CentralAir, train$PriceCat)
cat("\nTabla de contingencia:\n"); print(con_2x2)
cat("\nMarginales por fila:\n");   print(rowSums(con_2x2))
cat("\nMarginales por columna:\n");print(colSums(con_2x2))
cat("\nTabla con totales:\n");     print(addmargins(con_2x2))

cat("\nFrecuencias relativas conjuntas:\n")
print(round(prop.table(con_2x2), 4))
cat("\nFrecuencias relativas por fila (condicionadas por CentralAir):\n")

print(round(prop.table(con_2x2, margin = 1), 4))
cat("\nFrecuencias relativas por columna (condicionadas por PriceCat):\n")
print(round(prop.table(con_2x2, margin = 2), 4))

# Gráfico de mosaico (PDF sección 1.1.3)
mosaicplot(con_2x2,
           main = "Gráfico de Mosaico: CentralAir × PriceCat",
           sub  = "House Prices — Ames, Iowa",
           color = c("#74ADD1", "#F46D43"),
           xlab = "Aire Central", ylab = "Categoría de Precio")


# ==============================================================
# c.3) TEST DE INDEPENDENCIA: CHI-CUADRADO Y FISHER
# ==============================================================
# Referencia PDF sección 2

cat("\n", strrep("-", 55), "\n")
cat("--- c.3) Test de independencia ---\n")
cat(strrep("-", 55), "\n")

chi_2x2 <- suppressWarnings(chisq.test(con_2x2))

cat("\nFrecuencias esperadas bajo independencia:\n")
print(round(chi_2x2$expected, 2))

test_independencia(con_2x2, "CentralAir × PriceCat")


# ==============================================================
# c.4) TABLA N×M: Neighborhood × PriceCat
# ==============================================================

cat("\n", strrep("-", 55), "\n")
cat("--- c.4) Tabla de contingencia N×M: Neighborhood × PriceCat ---\n")
cat(strrep("-", 55), "\n")

top8 <- train %>% count(Neighborhood) %>% arrange(desc(n)) %>%
  slice_head(n = 8) %>% pull(Neighborhood)

con_NxM <- table(
  train$Neighborhood[train$Neighborhood %in% top8],
  train$PriceCat[train$Neighborhood %in% top8]
)
con_NxM <- con_NxM[rowSums(con_NxM) > 0, ]

cat("\nTabla con totales:\n"); print(addmargins(con_NxM))
cat("\nProporciones por fila (perfil del barrio):\n")
print(round(prop.table(con_NxM, margin = 1), 3))

mosaicplot(con_NxM,
           main = "Mosaico: Neighborhood (Top 8) × Categoría de Precio",
           color = c("#74ADD1", "#F46D43"), las = 2, cex.axis = 0.8)

chi_NxM <- suppressWarnings(chisq.test(con_NxM))
test_independencia(con_NxM, "Neighborhood × PriceCat")


# ==============================================================
# c.5) MEDIDAS DE ASOCIACIÓN: ESCALA NOMINAL
# ==============================================================
# Referencia PDF secciones 3.1 a 3.4

cat("\n", strrep("-", 55), "\n")
cat("--- c.5) Medidas de asociación — Escala Nominal ---\n")
cat(strrep("-", 55), "\n")

# --- Tabla 2×2: CentralAir × PriceCat ---
cat("\n===== Medidas para la tabla 2×2: CentralAir × PriceCat =====\n")

# C de Pearson (PDF sección 3.1) — función manual
Contingency_C <- function(cont) {
  chi <- chisq.test(cont)
  n <- sum(cont)
  unname(sqrt(chi$statistic / (chi$statistic + n)))
}
C_2x2 <- Contingency_C(con_2x2)
cat(sprintf("\n   C de Pearson (manual)    = %.4f\n", C_2x2))
C_2x2_dt <- ContCoef(con_2x2)
cat(sprintf("   C de Pearson (DescTools) = %.4f\n", C_2x2_dt))

# Phi (PDF sección 3.2) — función manual
PhiCoef <- function(x) unname(sqrt(chisq.test(x)$statistic / sum(x)))
phi_2x2 <- PhiCoef(con_2x2)
cat(sprintf("   Phi (manual)            = %.4f\n", phi_2x2))
phi_2x2_dt <- Phi(con_2x2)
cat(sprintf("   Phi (DescTools)         = %.4f\n", phi_2x2_dt))

# V de Cramer (PDF sección 3.3)
V_Cramer <- function(x) unname(sqrt(chisq.test(x)$statistic / (sum(x) * (min(dim(x)) - 1))))
v_2x2 <- V_Cramer(con_2x2)
cat(sprintf("   V de Cramer (manual)    = %.4f\n", v_2x2))
v_2x2_dt <- CramerV(con_2x2)
cat(sprintf("   V de Cramer (DescTools) = %.4f\n", v_2x2_dt))

# Q de Yule (PDF sección 3.4) — Solo tablas 2×2
a <- con_2x2[1,1]; b <- con_2x2[1,2]; c_val <- con_2x2[2,1]; d <- con_2x2[2,2]
Q_Yule <- (a*d - b*c_val) / (a*d + b*c_val)
cat(sprintf("   Q de Yule               = %.4f\n", Q_Yule))
cat("   Interpretación Q de Yule: ")
absQ <- abs(Q_Yule)
if (absQ <= 0.25) { cat("Asociación muy débil.\n")
} else if (absQ <= 0.50) { cat("Asociación débil.\n")
} else if (absQ <= 0.75) { cat("Asociación moderada.\n")
} else { cat("Asociación fuerte.\n") }

cat("\n   Resumen con assocstats (paquete vcd):\n")
print(assocstats(con_2x2))

# --- Tabla N×M: Neighborhood × PriceCat ---
cat("\n===== Medidas para la tabla N×M: Neighborhood × PriceCat =====\n")
C_NxM <- ContCoef(con_NxM)
v_NxM <- CramerV(con_NxM)
cat(sprintf("\n   C de Pearson  = %.4f\n", C_NxM))
cat(sprintf("   V de Cramer   = %.4f\n", v_NxM))
cat("\n   Resumen con assocstats:\n"); print(assocstats(con_NxM))


# ==============================================================
# c.6) MEDIDAS DE ASOCIACIÓN: ESCALA ORDINAL
# ==============================================================
# Referencia PDF secciones 4.1 a 4.4

cat("\n", strrep("-", 55), "\n")
cat("--- c.6) Medidas de asociación — Escala Ordinal ---\n")
cat(strrep("-", 55), "\n")

# --- Tabla ordinal 1: QualCat × AgeCat ---
con_ord <- table(train$QualCat, train$AgeCat)
cat("\nTabla de contingencia ordinal: Calidad × Antigüedad\n")
print(addmargins(con_ord))

mosaicplot(con_ord,
           main = "Mosaico: Calidad General × Antigüedad de la vivienda",
           color = c("#2C7BB6", "#74ADD1", "#F46D43", "#D73027"), las = 2, cex.axis = 0.75)

chi_ord <- suppressWarnings(chisq.test(con_ord))
test_independencia(con_ord, "QualCat × AgeCat")

# Gamma de Goodman y Kruskal (PDF sección 4.1)
gamma_result <- GoodmanKruskalGamma(con_ord, conf.level = 0.95)
cat("\nGamma de Goodman-Kruskal:\n"); print(gamma_result)
cat("   Interpretación Gamma: ")
gamma_val <- gamma_result[1]
if (abs(gamma_val) <= 0.25) { cat("Asociación muy débil.\n")
} else if (abs(gamma_val) <= 0.50) { cat("Asociación débil.\n")
} else if (abs(gamma_val) <= 0.75) { cat("Asociación moderada.\n")
} else { cat("Asociación fuerte.\n") }

# Tau-b de Kendall (PDF sección 4.3)
taub_result <- KendallTauB(con_ord, conf.level = 0.95)
cat("\nTau-b de Kendall:\n"); print(taub_result)

# Tau-c de Stuart/Kendall (PDF sección 4.4)
tauc_result <- StuartTauC(con_ord, conf.level = 0.95)
cat("\nTau-c de Stuart-Kendall:\n"); print(tauc_result)


# --- Tabla ordinal 2: OverallQual × KitchenQual (con D de Somers) ---
cat("\n--- Tabla ordinal 2: OverallQual (agrupada) × KitchenQual ---\n")

# Agrupamos OverallQual en 3 categorías (como en la teoría)
train$Qual_cat3 <- cut(as.integer(train$OverallQual),
                       breaks = c(0, 4, 7, 10),
                       labels = c("Baja (1-4)", "Media (5-7)", "Alta (8-10)"),
                       ordered_result = TRUE)

tab_qual_kit <- table(droplevels(train$Qual_cat3), droplevels(train$KitchenQual))
cat("Frecuencias absolutas:\n"); print(addmargins(tab_qual_kit))

chi_qk <- suppressWarnings(chisq.test(tab_qual_kit, correct = FALSE))
test_independencia(tab_qual_kit, "OverallQual × KitchenQual")

# Medidas ordinales incluyendo D de Somers 
gamma_qk  <- GoodmanKruskalGamma(tab_qual_kit, conf.level = 0.95)
somers_c  <- SomersDelta(tab_qual_kit, direction = "column", conf.level = 0.95)
somers_r  <- SomersDelta(tab_qual_kit, direction = "row",    conf.level = 0.95)
tau_b_qk  <- KendallTauB(tab_qual_kit, conf.level = 0.95)
tau_c_qk  <- StuartTauC(tab_qual_kit,  conf.level = 0.95)

cat("\nMedidas de asociación ordinal:\n")
cat(sprintf("  Gamma de Goodman-Kruskal : %.4f  [IC 95%%: %.4f – %.4f]\n",
            gamma_qk[1], gamma_qk[2], gamma_qk[3]))
cat(sprintf("  D de Somers (C|R)        : %.4f  [IC 95%%: %.4f – %.4f]\n",
            somers_c[1], somers_c[2], somers_c[3]))
cat(sprintf("  D de Somers (R|C)        : %.4f  [IC 95%%: %.4f – %.4f]\n",
            somers_r[1], somers_r[2], somers_r[3]))
cat(sprintf("  Tau-b de Kendall         : %.4f  [IC 95%%: %.4f – %.4f]\n",
            tau_b_qk[1], tau_b_qk[2], tau_b_qk[3]))
cat(sprintf("  Tau-c de Stuart-Kendall  : %.4f  [IC 95%%: %.4f – %.4f]\n",
            tau_c_qk[1], tau_c_qk[2], tau_c_qk[3]))

# Gráfico de barras apiladas proporcionales
df_qk <- as.data.frame(tab_qual_kit) %>% rename(Qual_cat = Var1, KitchenQual = Var2)
p_qk  <- ggplot(df_qk, aes(x = Qual_cat, y = Freq, fill = KitchenQual)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_brewer(palette = "RdYlGn", direction = 1, name = "Calidad cocina") +
  labs(title    = "Calidad General vs Calidad de Cocina",
       subtitle = "Distribución proporcional por categoría de calidad general",
       x = "Calidad general (agrupada)", y = "Proporción") + theme_hp
print(p_qk)


# ==============================================================
# c.7) TABLA TRIDIMENSIONAL Y TEST DE COCHRAN-MANTEL-HAENSZEL
# ==============================================================
# Referencia PDF sección 3.5

cat("\n", strrep("-", 55), "\n")
cat("--- c.7) Tabla tridimensional y Test Cochran-Mantel-Haenszel ---\n")
cat(strrep("-", 55), "\n")

con_3d <- xtabs(~ CentralAir + PriceCat + AgeCat, data = train)
cat("\nTabla tridimensional: CentralAir × PriceCat × AgeCat\n")
ftable(con_3d)

cmh_test <- mantelhaen.test(con_3d)
cat(sprintf("\nTest Cochran-Mantel-Haenszel:\n"))
cat(sprintf("   M² = %.4f, gl = %d, p-valor = %.4e\n",
            cmh_test$statistic, cmh_test$parameter, cmh_test$p.value))
cat(sprintf("   Odds Ratio común estimado = %.4f\n", cmh_test$estimate))

cat("\n   H0: CentralAir y PriceCat son independientes dado AgeCat.\n")
cat("   Conclusión: ")
if (cmh_test$p.value < 0.05) {
  cat("Rechazamos H0 → La asociación entre CentralAir y PriceCat\n")
  cat("   persiste incluso controlando por la antigüedad de la vivienda.\n")
} else {
  cat("No rechazamos H0 → La asociación desaparece al controlar por antigüedad.\n")
}

# Tablas parciales por estrato
cat("\n   Tablas parciales por estrato de antigüedad:\n")
for (i in seq_len(dim(con_3d)[3])) {
  cat(sprintf("\n   --- %s ---\n", dimnames(con_3d)[[3]][i]))
  print(con_3d[, , i])
}


# ==============================================================
# c.8) TABLA MIXTA: BldgType × QualCat (nominal × ordinal)
# ==============================================================
# Referencia PDF sección 4.5

cat("\n", strrep("-", 55), "\n")
cat("--- c.8) Asociación Nominal × Ordinal: BldgType × Calidad ---\n")
cat(strrep("-", 55), "\n")

con_mixta <- table(train$BldgType, train$QualCat)
cat("\nTabla de contingencia:\n"); print(addmargins(con_mixta))

chi_mixta <- suppressWarnings(chisq.test(con_mixta))
test_independencia(con_mixta, "BldgType × QualCat")
cat(sprintf("\nC de Pearson = %.4f\n", ContCoef(con_mixta)))
cat(sprintf("V de Cramer  = %.4f\n", CramerV(con_mixta)))
print(assocstats(con_mixta))

mosaicplot(con_mixta,
           main = "Mosaico: Tipo de Edificio × Calidad General",
           color = c("#2C7BB6", "#74ADD1", "#FDAE61", "#D7191C"), las = 2, cex.axis = 0.7)

cat("\n   Nota: Al tener una variable nominal (BldgType), las medidas ordinales\n")
cat("   (Gamma, Tau-b, Tau-c) no son estrictamente apropiadas.\n")


# ==============================================================
# c.9) RESUMEN GLOBAL DE ASOCIACIONES
# ==============================================================

cat("\n", strrep("-", 55), "\n")
cat("--- c.9) Resumen global de medidas de asociación ---\n")
cat(strrep("-", 55), "\n")

resumen_asoc <- data.frame(
  Par_de_Variables = c(
    "CentralAir × PriceCat (2×2)",
    "Neighborhood × PriceCat (N×2)",
    "QualCat × AgeCat (ordinal)",
    "OverallQual × KitchenQual (ordinal)",
    "BldgType × QualCat (nominal×ordinal)"
  ),
  Chi2 = c(
    round(chi_2x2$statistic, 2), round(chi_NxM$statistic, 2),
    round(chi_ord$statistic, 2), round(chi_qk$statistic, 2),
    round(chi_mixta$statistic, 2)
  ),
  p_valor = c(
    formatC(chi_2x2$p.value, format="e", digits=2),
    formatC(chi_NxM$p.value, format="e", digits=2),
    formatC(chi_ord$p.value, format="e", digits=2),
    formatC(chi_qk$p.value, format="e", digits=2),
    formatC(chi_mixta$p.value, format="e", digits=2)
  ),
  V_Cramer = c(
    round(v_2x2_dt, 4), round(v_NxM, 4),
    round(CramerV(con_ord), 4), round(CramerV(tab_qual_kit), 4),
    round(CramerV(con_mixta), 4)
  ),
  stringsAsFactors = FALSE
)
cat("\n"); print(resumen_asoc, row.names = FALSE)

cat("\n   Interpretación V de Cramer:\n")
cat("     0.00 - 0.10 → Asociación insignificante\n")
cat("     0.10 - 0.30 → Asociación débil\n")
cat("     0.30 - 0.50 → Asociación moderada\n")
cat("     0.50 - 1.00 → Asociación fuerte\n")

# Barplot comparativo de V de Cramer
par(mar = c(8, 4, 4, 2))
barplot(
  resumen_asoc$V_Cramer,
  names.arg = c("CentralAir\n×PriceCat", "Neighborhood\n×PriceCat",
                "Calidad\n×Antigüedad", "OverallQual\n×KitchenQual", "BldgType\n×Calidad"),
  col = ifelse(resumen_asoc$V_Cramer > 0.30, "#D7191C",
               ifelse(resumen_asoc$V_Cramer > 0.10, "#FDAE61", "#74ADD1")),
  main = "Comparación de la V de Cramer entre pares de variables",
  ylab = "V de Cramer", ylim = c(0, max(resumen_asoc$V_Cramer) * 1.2),
  las = 2, cex.names = 0.8
)
abline(h = 0.30, lty = 2, col = "gray40")
text(0.5, 0.32, "Moderada", col = "gray40", cex = 0.8, pos = 4)
abline(h = 0.10, lty = 3, col = "gray60")
text(0.5, 0.12, "Débil", col = "gray60", cex = 0.8, pos = 4)
par(mar = c(5, 4, 4, 2) + 0.1)


cat("\n", strrep("=", 65), "\n")
cat("✓ PARTE 1.5 — Tablas de Contingencia y Medidas de Asociación completada.\n")
cat(strrep("=", 65), "\n\n")


# ==============================================================
# PARTE 2: ANÁLISIS MULTIVARIANTE
# Secciones:
#   d) Análisis de Componentes Principales (PCA)
#   e) Análisis de Correspondencias (CA simple y MCA)
#   f) Análisis Clúster (K-means y Jerárquico)
# ==============================================================

pacman::p_load(
  FactoMineR, factoextra, ggrepel,
  cluster, dendextend, NbClust, RColorBrewer
)


# ################################################################
# ##   d) ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)             ##
# ################################################################

cat("\n", strrep("=", 65), "\n")
cat("SECCIÓN d) — ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)\n")
cat(strrep("=", 65), "\n")

# ------------------------------------------------------------------
# d.1) Selección y preparación de variables numéricas para PCA
# ------------------------------------------------------------------
pca_vars <- c(
  "LotFrontage", "LotArea",     "GrLivArea",   "TotalBsmtSF",
  "FlrSF_1st",   "FlrSF_2nd",   "GarageArea",  "WoodDeckSF",
  "OpenPorchSF", "TotalBaths",  "TotRmsAbvGrd", "Fireplaces",
  "HouseAge",    "RemodAge",    "BedroomAbvGr",  "KitchenAbvGr"
)

idx_pca   <- complete.cases(train[, pca_vars])
train_pca <- train[idx_pca, pca_vars]

cat(sprintf("\nVariables seleccionadas para PCA: %d\n", length(pca_vars)))
cat(sprintf("Observaciones válidas (sin NA)  : %d\n", nrow(train_pca)))

cat("\nMedias de las variables seleccionadas:\n")
print(round(apply(train_pca, 2, mean), 2))
cat("\nVarianzas de las variables seleccionadas:\n")
print(round(apply(train_pca, 2, var), 2))
cat("\nLas variables tienen escalas muy distintas → estandarizar (scale=TRUE).\n")


# ------------------------------------------------------------------
# d.2) Ejecución del PCA con prcomp
# ------------------------------------------------------------------
pca_out <- prcomp(train_pca, scale = TRUE, center = TRUE)

cat("\nCargas (loadings) de las 4 primeras componentes principales:\n")
print(round(pca_out$rotation[, 1:4], 3))


# ------------------------------------------------------------------
# d.3) Varianza explicada y selección del número de componentes
# ------------------------------------------------------------------
pca_var  <- pca_out$sdev^2
pve      <- pca_var / sum(pca_var)
pve_acum <- cumsum(pve)

tabla_pve <- data.frame(
  CP            = paste0("PC", seq_along(pve)),
  Autovalor     = round(pca_var, 4),
  PVE           = round(pve * 100, 2),
  PVE_Acumulada = round(pve_acum * 100, 2)
)
cat("\n--- Varianza explicada por cada componente principal ---\n")
print(tabla_pve)

n_kaiser <- sum(pca_var > 1)
n_80     <- which(pve_acum >= 0.80)[1]
cat(sprintf("\nCriterio de Kaiser: retener %d componentes (autovalor > 1)\n", n_kaiser))
cat(sprintf("Criterio del 80%% de varianza: retener %d componentes (PVE acum. = %.1f%%)\n",
            n_80, pve_acum[n_80]*100))


# ------------------------------------------------------------------
# d.4) Scree Plot
# ------------------------------------------------------------------
p_scree <- fviz_eig(pca_out, addlabels = TRUE, ylim = c(0, 40),
                    barfill = "#2C7BB6", barcolor = "#2C7BB6",
                    linecolor = "#D7191C") +
  geom_hline(yintercept = 100/length(pca_vars), linetype = "dashed",
             color = "#1A9641", linewidth = 0.8) +
  labs(
    title    = "Scree Plot — Porcentaje de varianza explicada por CP",
    subtitle = sprintf("Kaiser: %d CPs | 80%% varianza: %d CPs", n_kaiser, n_80),
    x = "Componente Principal", y = "% Varianza Explicada"
  ) + theme_hp
print(p_scree)

# PVE individual y acumulada (base R)
par(mfrow = c(1, 2))
plot(pve, type = "b", pch = 19, col = "#2C7BB6",
     xlab = "CP", ylab = "PVE", main = "PVE por componente", ylim = c(0, max(pve)+0.05))
abline(h = 1/length(pca_vars), col = "#1A9641", lty = 2)
plot(pve_acum, type = "b", pch = 19, col = "#D7191C",
     xlab = "CP", ylab = "PVE Acumulada", main = "PVE Acumulada", ylim = c(0, 1))
abline(h = 0.80, col = "#1A9641", lty = 2, lwd = 2)
par(mfrow = c(1, 1))


# ------------------------------------------------------------------
# d.5) Biplot: observaciones y variables en PC1-PC2
# ------------------------------------------------------------------
p_biplot <- fviz_pca_biplot(pca_out, repel = TRUE,
                            col.var = "#D7191C", col.ind = "#2C7BB6",
                            alpha.ind = 0.15, geom.ind = "point",
                            pointsize = 1.2, arrowsize = 0.8, labelsize = 3) +
  labs(
    title    = "Biplot PCA — Observaciones y Variables (PC1 vs PC2)",
    subtitle = sprintf("PC1 = %.1f%% + PC2 = %.1f%% → %.1f%% varianza explicada",
                       pve[1]*100, pve[2]*100, (pve[1]+pve[2])*100)
  ) + theme_hp
print(p_biplot)


# ------------------------------------------------------------------
# d.6) Individuos coloreados por OverallQual
# ------------------------------------------------------------------
qual_grupo <- factor(train$OverallQual[idx_pca], ordered = TRUE)

p_biplot_qual <- fviz_pca_ind(
  pca_out, geom = "point", habillage = qual_grupo,
  palette = viridis(10, option = "plasma"),
  pointsize = 1.5, alpha.ind = 0.5, addEllipses = FALSE
) +
  labs(title    = "PCA — Observaciones coloreadas por Calidad General",
       subtitle = "OverallQual como variable suplementaria cualitativa",
       color    = "Calidad\nGeneral") + theme_hp
print(p_biplot_qual)


# ------------------------------------------------------------------
# d.7) Contribución de variables y círculo de correlaciones
# ------------------------------------------------------------------
p_contrib1 <- fviz_contrib(pca_out, choice = "var", axes = 1, top = 16,
                           fill = "#2C7BB6", color = "#2C7BB6") +
  labs(title = "Contribución de variables a PC1") + theme_hp
p_contrib2 <- fviz_contrib(pca_out, choice = "var", axes = 2, top = 16,
                           fill = "#D7191C", color = "#D7191C") +
  labs(title = "Contribución de variables a PC2") + theme_hp
grid.arrange(p_contrib1, p_contrib2, ncol = 2,
             top = "Contribución de las variables a las dos primeras CP")

p_var_pca <- fviz_pca_var(pca_out, col.var = "contrib",
                          gradient.cols = c("#FEE090", "#D7191C"),
                          repel = TRUE, labelsize = 3.5) +
  labs(title    = "Círculo de correlaciones — Variables en PC1-PC2",
       subtitle = "Coloreado por contribución a las dos primeras CP") + theme_hp
print(p_var_pca)


# ------------------------------------------------------------------
# d.8) Correlación entre CP y SalePrice
# ------------------------------------------------------------------
scores_pca <- pca_out$x[, 1:n_kaiser]
precio_pca <- train$SalePrice[idx_pca]

cor_cp_precio <- cor(scores_pca, precio_pca, use = "pairwise.complete.obs")
cat("\nCorrelación de las componentes principales con SalePrice:\n")
print(round(cor_cp_precio, 4))

# Scatter PC1 vs SalePrice
p_pc1_sp <- data.frame(PC1 = pca_out$x[, 1],
                       SalePrice = train$SalePrice[idx_pca],
                       OverallQual = qual_grupo) %>%
  ggplot(aes(x = PC1, y = SalePrice, color = OverallQual)) +
  geom_point(alpha = 0.45, size = 1.5) +
  geom_smooth(method = "lm", se = FALSE, color = "black",
              linewidth = 0.8, linetype = "dashed") +
  scale_y_continuous(labels = label_dollar()) +
  labs(title    = "Relación entre PC1 y SalePrice",
       subtitle = "La primera componente resume la estructura física de la vivienda",
       x = "PC1", y = "SalePrice (USD)", color = "OverallQual") + theme_hp
print(p_pc1_sp)


# ------------------------------------------------------------------
# d.9) PCA con FactoMineR (análisis complementario)
# ------------------------------------------------------------------
pca_FM <- PCA(train_pca, scale.unit = TRUE, ncp = n_kaiser, graph = FALSE)

cat("\n--- Autovalores (FactoMineR) ---\n")
print(round(pca_FM$eig, 3))
cat("\n--- Coordenadas de las variables (2 primeras dim.) ---\n")
print(round(pca_FM$var$coord[, 1:2], 3))
cat("\n--- Calidad de representación (cos2) de las variables ---\n")
print(round(pca_FM$var$cos2[, 1:2], 3))



# ################################################################
# ##   e) ANÁLISIS DE CORRESPONDENCIAS                          ##
# ################################################################

cat("\n", strrep("=", 65), "\n")
cat("SECCIÓN e) — ANÁLISIS DE CORRESPONDENCIAS\n")
cat(strrep("=", 65), "\n")

# ==================================================================
# e.1) ANÁLISIS DE CORRESPONDENCIAS SIMPLES (CA)
# ==================================================================

cat("\n--- e.1) Análisis de Correspondencias Simples (CA) ---\n")

# --- e.1.1) Tabla 1: Neighborhood × Grupo de Calidad ---
top_neighs <- train %>% count(Neighborhood) %>% arrange(desc(n)) %>%
  slice_head(n = 10) %>% pull(Neighborhood)

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

tabla_cont_1 <- table(train_ca$Neighborhood, train_ca$QualGrupo)
cat("\nTabla de contingencia: Neighborhood × Grupo de Calidad\n")
print(tabla_cont_1)

chi_test_1 <- test_independencia(tabla_cont_1, "Neighborhood × Grupo de Calidad")

res_ca1 <- CA(tabla_cont_1, graph = FALSE)
eig_ca1 <- get_eigenvalue(res_ca1)
cat("\n--- Autovalores del CA ---\n"); print(round(eig_ca1, 3))

p_scree_ca1 <- fviz_screeplot(res_ca1, addlabels = TRUE, ylim = c(0, 100)) +
  labs(title = "Scree Plot — CA: Neighborhood × Calidad",
       subtitle = "% de inercia explicada por dimensión") + theme_hp
print(p_scree_ca1)

p_ca1_biplot <- fviz_ca_biplot(res_ca1, repel = TRUE,
                               col.row = "#2C7BB6", col.col = "#D7191C",
                               shape.row = 17, shape.col = 15, labelsize = 4) +
  labs(title    = "CA Biplot simétrico: Neighborhood × Grupo de Calidad",
       subtitle = sprintf("Dim.1 = %.1f%% + Dim.2 = %.1f%% → %.1f%% inercia explicada",
                          eig_ca1[1,2], eig_ca1[2,2], eig_ca1[1,3])) + theme_hp
print(p_ca1_biplot)

p_ca1_row <- fviz_contrib(res_ca1, choice = "row", axes = 1:2, top = 10, fill = "#2C7BB6") +
  labs(title = "Contribución de los barrios a Dim.1-2") + theme_hp
p_ca1_col <- fviz_contrib(res_ca1, choice = "col", axes = 1:2, fill = "#D7191C") +
  labs(title = "Contribución de los niveles de calidad a Dim.1-2") + theme_hp
grid.arrange(p_ca1_row, p_ca1_col, ncol = 2,
             top = "Contribuciones al Análisis de Correspondencias Simple")


# --- e.1.2) Tabla 2: BldgType × SaleCondition ---
cat("\n--- Segundo CA: BldgType × SaleCondition ---\n")
tabla_cont_2 <- table(train$BldgType, train$SaleCondition)
print(tabla_cont_2)

chi_test_2 <- test_independencia(tabla_cont_2, "BldgType × SaleCondition")

res_ca2 <- CA(tabla_cont_2, graph = FALSE)
p_ca2_biplot <- fviz_ca_biplot(res_ca2, repel = TRUE, map = "rowprincipal",
                               col.row = "#2C7BB6", col.col = "#D7191C", labelsize = 4) +
  labs(title    = "CA Biplot asimétrico: BldgType × SaleCondition",
       subtitle = "Mapa de filas en coordenadas principales") + theme_hp
print(p_ca2_biplot)


# ==================================================================
# e.2) ANÁLISIS DE CORRESPONDENCIAS MÚLTIPLES (MCA)
# ==================================================================

cat("\n--- e.2) Análisis de Correspondencias Múltiples (MCA) ---\n")

mca_vars_activas   <- c("MSZoning", "BldgType", "HouseStyle", "Foundation",
                        "CentralAir", "GarageType", "SaleCondition")
mca_vars_quanti_sup <- c("SalePrice", "GrLivArea", "HouseAge")
mca_vars_quali_sup  <- c("OverallQual")

train_mca <- train %>%
  dplyr::select(all_of(c(mca_vars_activas, mca_vars_quali_sup, mca_vars_quanti_sup))) %>%
  drop_na()

idx_quali_sup  <- which(names(train_mca) %in% mca_vars_quali_sup)
idx_quanti_sup <- which(names(train_mca) %in% mca_vars_quanti_sup)

cat(sprintf("Variables activas: %d | Cuanti. sup.: %d | Cuali. sup.: %d | Obs.: %d\n",
            length(mca_vars_activas), length(mca_vars_quanti_sup),
            length(mca_vars_quali_sup), nrow(train_mca)))

res_mca <- MCA(train_mca, quali.sup = idx_quali_sup,
               quanti.sup = idx_quanti_sup, graph = FALSE)

eig_mca <- get_eigenvalue(res_mca)
cat("\n--- Autovalores del MCA (primeras 10 dimensiones) ---\n")
print(round(head(eig_mca, 10), 3))

p_scree_mca <- fviz_screeplot(res_mca, addlabels = TRUE, ncp = 15, ylim = c(0, 20)) +
  labs(title = "Scree Plot — MCA: Variables categóricas de vivienda",
       subtitle = "% de inercia explicada por dimensión") + theme_hp
print(p_scree_mca)

# Mapa de categorías activas
p_mca_var <- fviz_mca_var(res_mca, repel = TRUE, col.var = "contrib",
                          gradient.cols = c("#FEE090", "#FC8D59", "#D7191C"),
                          labelsize = 3, ggtheme = theme_hp) +
  labs(title = "MCA — Mapa de categorías (variables activas)",
       subtitle = "Coloreado por contribución a las dos primeras dimensiones")
print(p_mca_var)

# Individuos coloreados por OverallQual
p_mca_ind <- fviz_mca_ind(res_mca, geom = "point", habillage = idx_quali_sup,
                          addEllipses = TRUE, ellipse.level = 0.75,
                          pointsize = 1, alpha.ind = 0.3, ggtheme = theme_hp) +
  labs(title = "MCA — Individuos coloreados por Calidad General",
       subtitle = "OverallQual como variable cualitativa suplementaria (elipses al 75%)")
print(p_mca_ind)

# Biplot MCA
p_mca_biplot <- fviz_mca_biplot(res_mca, repel = TRUE, geom.ind = "point",
                                col.ind = "gray70", alpha.ind = 0.15,
                                col.var = "contrib",
                                gradient.cols = c("#FEE090", "#D7191C"),
                                labelsize = 3, ggtheme = theme_hp) +
  labs(title = "MCA — Biplot: individuos y categorías de variables",
       subtitle = "Las categorías cercanas comparten perfiles similares")
print(p_mca_biplot)

# Contribuciones por dimensión
p_mca_c1 <- fviz_contrib(res_mca, choice = "var", axes = 1, top = 15, fill = "#2C7BB6") +
  labs(title = "Contribución de categorías a Dim.1") + theme_hp
p_mca_c2 <- fviz_contrib(res_mca, choice = "var", axes = 2, top = 15, fill = "#D7191C") +
  labs(title = "Contribución de categorías a Dim.2") + theme_hp
grid.arrange(p_mca_c1, p_mca_c2, ncol = 2,
             top = "Contribuciones al Análisis de Correspondencias Múltiples")

# Descripción de dimensiones
desc_mca <- dimdesc(res_mca, axes = c(1, 2))
cat("\n--- Descripción de la Dimensión 1 del MCA ---\n"); print(desc_mca[[1]])
cat("\n--- Descripción de la Dimensión 2 del MCA ---\n"); print(desc_mca[[2]])

# Variables cuantitativas suplementarias proyectadas
p_mca_quanti <- fviz_mca_var(res_mca, choice = "quanti.sup",
                             repel = TRUE, ggtheme = theme_hp) +
  labs(title = "MCA — Variables cuantitativas suplementarias",
       subtitle = "Correlación de SalePrice, GrLivArea y HouseAge con las dimensiones")
print(p_mca_quanti)



# ################################################################
# ##   f) ANÁLISIS CLÚSTER                                      ##
# ################################################################

cat("\n", strrep("=", 65), "\n")
cat("SECCIÓN f) — ANÁLISIS CLÚSTER\n")
cat(strrep("=", 65), "\n")

# ------------------------------------------------------------------
# f.0) Preparación de datos
# ------------------------------------------------------------------
clust_vars <- c("GrLivArea", "TotalBsmtSF", "GarageArea", "LotArea",
                "TotalBaths", "HouseAge", "TotRmsAbvGrd", "Fireplaces")

train_clust <- train %>% dplyr::select(all_of(clust_vars)) %>% drop_na()
train_clust_scaled <- scale(train_clust)
idx_clust_valid <- which(complete.cases(train[, clust_vars]))

cat(sprintf("\nVariables para clustering: %d\n", length(clust_vars)))
cat(sprintf("Observaciones: %d\n", nrow(train_clust_scaled)))


# ==================================================================
# f.1) K-MEANS: selección del número óptimo
# ==================================================================

cat("\n--- f.1) K-Means Clustering ---\n")

# Método del codo y silueta 
p_elbow <- fviz_nbclust(train_clust_scaled, kmeans, method = "wss",
                        k.max = 10, nstart = 50) +
  labs(title = "Método del codo — Suma de cuadrados intra-clúster (WSS)",
       subtitle = "El 'codo' sugiere el número óptimo de clústeres") + theme_hp
print(p_elbow)

p_silhouette <- fviz_nbclust(train_clust_scaled, kmeans, method = "silhouette",
                             k.max = 10, nstart = 50) +
  labs(title = "Método de la silueta — Anchura media de silueta",
       subtitle = "Mayor anchura = mejor separación entre clústeres") + theme_hp
print(p_silhouette)

# Selección data-driven de K 
dist_clust <- dist(train_clust_scaled)
sil_prom <- sapply(2:8, function(k) {
  km_tmp <- kmeans(train_clust_scaled, centers = k, nstart = 50)
  mean(silhouette(km_tmp$cluster, dist_clust)[, 3])
})
k_opt <- (2:8)[which.max(sil_prom)]
cat(sprintf("\nK óptimo (silueta máxima): K = %d\n", k_opt))


# ------------------------------------------------------------------
# f.2) Comparación K=3, K=4, K=5 
# ------------------------------------------------------------------

km3 <- kmeans(train_clust_scaled, centers = 3, nstart = 50)
km4 <- kmeans(train_clust_scaled, centers = 4, nstart = 50)
km5 <- kmeans(train_clust_scaled, centers = 5, nstart = 50)

cat("\n--- Resultados K-Means ---\n")
cat(sprintf("K=3: BSS/TSS = %.1f%%\n", km3$betweenss/km3$totss*100))
cat(sprintf("K=4: BSS/TSS = %.1f%%\n", km4$betweenss/km4$totss*100))
cat(sprintf("K=5: BSS/TSS = %.1f%%\n", km5$betweenss/km5$totss*100))

# Visualización comparativa
p_km3 <- fviz_cluster(km3, data = train_clust_scaled,
                      geom = "point", pointsize = 1, alpha = 0.4,
                      ellipse.type = "convex",
                      palette = c("#2C7BB6", "#D7191C", "#1A9641"),
                      ggtheme = theme_hp) +
  labs(title = "K = 3", subtitle = sprintf("BSS/TSS = %.1f%%", km3$betweenss/km3$totss*100))

p_km4 <- fviz_cluster(km4, data = train_clust_scaled,
                      geom = "point", pointsize = 1, alpha = 0.4,
                      ellipse.type = "convex",
                      palette = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61"),
                      ggtheme = theme_hp) +
  labs(title = "K = 4", subtitle = sprintf("BSS/TSS = %.1f%%", km4$betweenss/km4$totss*100))

p_km5 <- fviz_cluster(km5, data = train_clust_scaled,
                      geom = "point", pointsize = 1, alpha = 0.4,
                      ellipse.type = "convex", ggtheme = theme_hp) +
  labs(title = "K = 5", subtitle = sprintf("BSS/TSS = %.1f%%", km5$betweenss/km5$totss*100))

grid.arrange(p_km3, p_km4, p_km5, ncol = 3,
             top = "Comparación de agrupamientos K-Means (K = 3, 4, 5)")

# Elegimos K=4 (consistente con codo y resultados habituales del dataset)
km_final <- km4

# Gráfico de silueta para K final
sil_km <- silhouette(km_final$cluster, dist_clust)
p_sil <- fviz_silhouette(sil_km, palette = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61")) +
  labs(title    = sprintf("Gráfico de silueta — K-Means (K = %d)", 4),
       subtitle = sprintf("Anchura media de silueta: %.3f", mean(sil_km[, 3]))) + theme_hp
print(p_sil)


# ==================================================================
# f.3) CLÚSTER JERÁRQUICO
# ==================================================================

cat("\n--- f.3) Clúster Jerárquico ---\n")

# Usamos una muestra para dendrograma legible
n_sample <- 200
idx_sample <- sample(nrow(train_clust_scaled), n_sample)
clust_sample <- train_clust_scaled[idx_sample, ]
dist_eucl <- dist(clust_sample, method = "euclidean")

# Comparación de métodos de enlace 
hc_complete <- hclust(dist_eucl, method = "complete")
hc_average  <- hclust(dist_eucl, method = "average")
hc_single   <- hclust(dist_eucl, method = "single")
hc_ward     <- hclust(dist_eucl, method = "ward.D2")

par(mfrow = c(1, 3))
plot(hc_complete, labels = FALSE, hang = -1, cex = 0.5,
     main = "Vinculación Completa", xlab = "", sub = "")
plot(hc_average, labels = FALSE, hang = -1, cex = 0.5,
     main = "Vinculación Promedio", xlab = "", sub = "")
plot(hc_ward, labels = FALSE, hang = -1, cex = 0.5,
     main = "Método de Ward (D2)", xlab = "", sub = "")
par(mfrow = c(1, 1))

# Coeficiente cofenético
coph_complete <- cor(cophenetic(hc_complete), dist_eucl)
coph_average  <- cor(cophenetic(hc_average),  dist_eucl)
coph_single   <- cor(cophenetic(hc_single),   dist_eucl)
coph_ward     <- cor(cophenetic(hc_ward),     dist_eucl)

cat("\nCoeficientes cofenéticos:\n")
cat(sprintf("  Completa : %.4f\n", coph_complete))
cat(sprintf("  Promedio : %.4f\n", coph_average))
cat(sprintf("  Individual: %.4f\n", coph_single))
cat(sprintf("  Ward (D2): %.4f\n", coph_ward))

metodos_coph <- c(Complete = coph_complete, Average = coph_average,
                  Single = coph_single, Ward = coph_ward)
mejor_metodo <- names(which.max(metodos_coph))
cat(sprintf("\nMejor método: %s (%.4f)\n", mejor_metodo, max(metodos_coph)))

# Dendrograma con corte K=4 (Ward, con dendextend)
dend_ward <- as.dendrogram(hc_ward)
dend_ward <- color_branches(dend_ward, k = 4,
                            col = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61"))
par(mar = c(2, 4, 3, 1))
plot(dend_ward, main = "Dendrograma — Método de Ward (n = 200, K = 4)",
     ylab = "Altura (distancia)", leaflab = "none")
cutree_height <- mean(c(sort(hc_ward$height, decreasing = TRUE)[3],
                        sort(hc_ward$height, decreasing = TRUE)[4]))
abline(h = cutree_height, col = "gray40", lty = 2, lwd = 1.5)
text(10, cutree_height + 0.5, "Corte K=4", col = "gray40", cex = 0.9)
par(mar = c(5, 4, 4, 2) + 0.1)

clust_hc <- cutree(hc_ward, k = 4)
cat("\nDistribución de observaciones en clústeres jerárquicos (K=4):\n")
print(table(clust_hc))

# Visualización en plano PCA
p_hc_pca <- fviz_cluster(list(data = clust_sample, cluster = clust_hc),
                         geom = "point", pointsize = 1.5, alpha = 0.5,
                         ellipse.type = "convex",
                         palette = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61"),
                         ggtheme = theme_hp) +
  labs(title = "Clúster Jerárquico (Ward, K=4) — Proyección en plano PCA",
       subtitle = sprintf("Muestra de %d observaciones | Distancia euclídea", n_sample))
print(p_hc_pca)


# ==================================================================
# f.4) PERFILADO Y COMPARACIÓN DE CLÚSTERES
# ==================================================================

cat("\n--- f.4) Perfilado de clústeres (K-Means, K=4) ---\n")

train$Cluster_KM <- NA
train$Cluster_KM[idx_clust_valid] <- km_final$cluster
train$Cluster_KM <- factor(train$Cluster_KM)

# Estadísticos descriptivos por clúster
perfil_clust <- train %>%
  filter(!is.na(Cluster_KM)) %>%
  group_by(Cluster_KM) %>%
  summarise(
    N              = n(),
    Precio_Med     = round(median(SalePrice)),
    Precio_Mean    = round(mean(SalePrice)),
    GrLivArea_Med  = round(median(GrLivArea)),
    TotalBsmtSF_Med = round(median(TotalBsmtSF)),
    GarageArea_Med = round(median(GarageArea)),
    LotArea_Med    = round(median(LotArea)),
    HouseAge_Med   = round(median(HouseAge)),
    TotalBaths_Med = round(median(TotalBaths), 1),
    .groups = "drop"
  )
cat("\nPerfil de cada clúster (medianas):\n"); print(perfil_clust)

# Boxplot: SalePrice por clúster
p_clust_price <- train %>%
  filter(!is.na(Cluster_KM)) %>%
  ggplot(aes(x = Cluster_KM, y = SalePrice, fill = Cluster_KM)) +
  geom_boxplot(outlier.alpha = 0.3, alpha = 0.7) +
  scale_y_continuous(labels = label_dollar()) +
  scale_fill_manual(values = c("#2C7BB6", "#D7191C", "#1A9641", "#FDAE61")) +
  labs(title = "Distribución del precio de venta por clúster (K-Means, K=4)",
       x = "Clúster", y = "Precio de venta (USD)") +
  theme_hp + theme(legend.position = "none")
print(p_clust_price)

# Heatmap de centroides estandarizados
centroides_df <- as.data.frame(km_final$centers)
centroides_df$Cluster <- paste0("C", 1:4)
centroides_long <- centroides_df %>%
  pivot_longer(cols = -Cluster, names_to = "Variable", values_to = "Valor_Z")

p_heatmap <- ggplot(centroides_long,
                    aes(x = Variable, y = Cluster, fill = Valor_Z)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = round(Valor_Z, 2)), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "#2C7BB6", mid = "white", high = "#D7191C",
                       midpoint = 0, name = "Z-Score") +
  labs(title = "Heatmap de centroides estandarizados (K-Means, K=4)",
       subtitle = "Rojo = por encima de la media; Azul = por debajo",
       x = NULL, y = "Clúster") +
  theme_hp + theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
                   panel.grid = element_blank())
print(p_heatmap)

# Distribución de OverallQual por clúster
p_clust_qual <- train %>%
  filter(!is.na(Cluster_KM)) %>%
  ggplot(aes(x = Cluster_KM, fill = OverallQual)) +
  geom_bar(position = "fill", alpha = 0.85) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_viridis_d(option = "plasma", name = "Calidad\nGeneral") +
  labs(title = "Composición de calidad general por clúster",
       x = "Clúster", y = "Proporción") + theme_hp
print(p_clust_qual)

# Tabla cruzada: Clúster × OverallQual con Chi-cuadrado
tabla_clust_qual <- table(train$Cluster_KM[!is.na(train$Cluster_KM)],
                          train$OverallQual[!is.na(train$Cluster_KM)])
cat("\nTabla cruzada: Clúster K-Means × Calidad General\n")
print(tabla_clust_qual)

chi_clust <- test_independencia(tabla_clust_qual, "Clúster K-Means × Calidad General")
cat("Conclusión adicional: Los clústeres están fuertemente asociados con la calidad general.\n")


# ==============================================================
# RESUMEN FINAL
# ==============================================================
cat("\n", strrep("=", 65), "\n")
cat("✓ ANÁLISIS COMPLETO FINALIZADO CON ÉXITO.\n")
cat(strrep("=", 65), "\n")
cat("\nResumen de resultados:\n")
cat(sprintf("  Preprocesado : %d obs. | %d variables\n", nrow(train), ncol(train)))
cat(sprintf("  PCA          : Kaiser = %d CPs | 80%% varianza = %d CPs\n", n_kaiser, n_80))
cat("  CA simple    : Asociación significativa Neighborhood × Calidad\n")
cat(sprintf("  MCA          : %d variables categóricas activas\n", length(mca_vars_activas)))
cat(sprintf("  K-Means      : K = 4 | BSS/TSS = %.1f%%\n", km_final$betweenss/km_final$totss*100))
cat(sprintf("  Jerárquico   : Mejor método = %s (cofenético = %.3f)\n",
            mejor_metodo, max(metodos_coph)))
cat(strrep("=", 65), "\n")
