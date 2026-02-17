
# ==============================================================================
# CARGA DE LIBRERÍAS
# ==============================================================================

if (!require("pacman")) install.packages("pacman")
library(pacman)

# Cargamos librerías de Práctica 1 y 2
p_load(tidyverse, VIM, moments, plotly, editrules, fastDummies, nortest, car)

# Ajustamos el directorio de trabajo
setwd("C:/Users/noefu/OneDrive/0. Inbox/2. ADAT/Mis prácticas en R/Titanic")



# ==============================================================================
# A) PREPARACIÓN DE LOS DATOS
# ==============================================================================


# 1. Cargar datos y Descripción Inicial
# ------------------------------------------------------------------------------

# Leemos los datos
datos <- read.table("titanic.txt", header=TRUE, sep=";", stringsAsFactors = FALSE)
df <- data.frame(datos)

# Descripción del conjunto
cat("Dimensiones:", nrow(df), "individuos y", ncol(df), "variables.\n")
head(df)



# 2. Transformación y Limpieza
# ------------------------------------------------------------------------------

# Corrección de formatos
df$fare <- as.numeric(gsub(",", ".", df$fare)) # Sustituimos coma por punto y pasamos a numérico
df$age  <- as.numeric(df$age)
df[df == ""] <- NA # Vacíos a NA

# Eliminación de variables con exceso de NAs
# Según la Práctica 2, si faltan demasiados datos y no aportan, se pueden quitar
summary(df)

df <- df %>% select(-body, -boat, -home.dest)

colSums(is.na(df))

# Imputación de NAs residuales (fare y embarked) 
df$fare[is.na(df$fare)] <- median(df$fare, na.rm = TRUE) # Mediana
df$embarked[is.na(df$embarked)] <- "S" # Moda

# Conversión a Factores (para gráficas y análisis)
df$pclass   <- factor(df$pclass, levels = c("1", "2", "3"), labels = c("Primera", "Segunda", "Tercera"))
df$survived <- factor(df$survived, levels = c("0", "1"), labels = c("No", "Sí"))
df$sex      <- as.factor(df$sex)
df$embarked <- as.factor(df$embarked)




# 3. Imputación Avanzada (k-NN para Age)
# ------------------------------------------------------------------------------

# Práctica 2: k-NN mantiene mejor la estructura que la media
# Usamos df (que ya tiene fare y embarked limpios)
df_imputed <- kNN(df, variable = c("age"), k = 5)

# Nos quedamos solo con las columnas originales (VIM añade más columnas)
df_clean <- df_imputed[, 1:ncol(df)]






# 4. Tratamiento de Outliers (Fare)
# ------------------------------------------------------------------------------

# Detección visual
boxplot(df_clean$fare, main = "Outliers en Fare (Antes)")

# Transformación Logarítmica para reducir asimetría 
df_clean$fare_log <- log(df_clean$fare + 1)  # +1 para evitar log(0)

# Comprobación de asimetría
cat("Fare Original:", skewness(df_clean$fare), "\n")
cat("Fare Log:", skewness(df_clean$fare_log), " (Más cercano a 0 es mejor)\n")





# 5. One Hot Encoding (Opcional para gráficos, necesario para modelos)
# ------------------------------------------------------------------------------

# Creamos dummies
df_dummies <- dummy_cols(df_clean, select_columns = c("sex", "embarked"), 
                         remove_first_dummy = TRUE)

# Dataset final para Análisis (EDA)
df_analysis <- df_clean 

# Resumen final del conjunto
summary(df_analysis)
cat("NAs restantes:", sum(is.na(df_analysis)), "\n")






# ==============================================================================
# B) ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. ANÁLISIS UNIVARIANTE (Distribuciones Individuales)
# ------------------------------------------------------------------------------

# --- Variable Numérica: EDAD ---
# [cite_start]Histograma con Curva de Densidad [cite: 633, 646]
p1 <- ggplot(df_analysis, aes(x = age)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "white") +
  geom_density(alpha = 0.4, fill = "red") +
  theme_minimal() +
  labs(title = "Distribución de la Edad", 
       subtitle = "La imputación k-NN ha preservado la forma natural",
       x = "Edad (Años)", y = "Densidad")
print(p1)

# --- Variable Numérica: TARIFA (Efecto de la Transformación) ---
# Comparativa: Original vs Log
p2 <- ggplot(df_analysis, aes(x = fare_log)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribución de la Tarifa (Transformación Log)", 
       subtitle = "Se ha corregido la asimetría extrema de los precios altos",
       x = "Log(Tarifa)", y = "Frecuencia")
print(p2)

# --- Variable Categórica: SUPERVIVENCIA ---
# Barplot simple con etiquetas
p3 <- ggplot(df_analysis, aes(x = survived, fill = survived)) +
  geom_bar() +
  scale_fill_manual(values = c("firebrick", "forestgreen")) +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  theme_minimal() +
  labs(title = "Conteo Total de Supervivientes", x = "¿Sobrevivió?", y = "Pasajeros")
print(p3)


# ------------------------------------------------------------------------------
# 2. ANÁLISIS BIVARIANTE (Relaciones clave)
# ------------------------------------------------------------------------------

# --- CLASE vs SUPERVIVENCIA (Barras Apiladas) ---
# [cite_start]Usamos position="fill" para ver porcentajes [cite: 108]
p4 <- ggplot(df_analysis, aes(x = pclass, fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Tasa de Supervivencia por Clase", 
       subtitle = "Clara correlación: A mejor clase, mayor supervivencia",
       x = "Clase", y = "Porcentaje") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")
print(p4)

# --- EDAD vs SUPERVIVENCIA (Boxplot + Jitter) ---
# [cite_start]Boxplot para ver medianas, Jitter para ver la densidad de puntos [cite: 615]
p5 <- ggplot(df_analysis, aes(x = survived, y = age, fill = survived)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) + # Ocultamos outliers del boxplot
  geom_jitter(width = 0.2, alpha = 0.1) +         # Los mostramos con jitter
  labs(title = "Distribución de Edad según Supervivencia", 
       subtitle = "Los niños (puntos bajos) tienen mayor presencia en 'Sí'",
       x = "¿Sobrevivió?", y = "Edad") +
  theme_minimal()
print(p5)

# --- TAMAÑO FAMILIA vs SUPERVIVENCIA ---
# Feature Engineering visual (Idea de tu amigo, muy valiosa)
p6 <- ggplot(df_analysis, aes(x = as.factor(family_size), fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Supervivencia según Tamaño Familiar", 
       x = "Miembros de la Familia (SibSp + Parch + 1)", y = "Porcentaje") +
  theme_minimal()
print(p6)


# ------------------------------------------------------------------------------
# 3. ANÁLISIS MULTIVARIANTE (Paneles / Facets)
# ------------------------------------------------------------------------------

# --- EL GRÁFICO DEFINITIVO: Clase + Sexo + Supervivencia ---
# [cite_start]Facet Wrap para separar por Sexo [cite: 300]
# Demuestra el protocolo "Mujeres y niños primero" dentro de cada clase
p7 <- ggplot(df_analysis, aes(x = pclass, fill = survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~sex) + 
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Supervivencia por Clase y Sexo",
       subtitle = "Casi todas las mujeres de 1ª y 2ª clase sobrevivieron",
       x = "Clase", y = "Porcentaje") +
  theme_minimal() +
  scale_fill_manual(values = c("gray40", "dodgerblue"))
print(p7)

# --- DENSIDAD MULTIVARIANTE ---
# Edad vs Supervivencia separado por Clase
p8 <- ggplot(df_analysis, aes(x = age, fill = survived)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~pclass) +
  labs(title = "Densidad de Edad por Clase y Supervivencia",
       subtitle = "En 3ª clase, incluso los niños tuvieron dificultades",
       x = "Edad", y = "Densidad") +
  theme_minimal()
print(p8)


# ------------------------------------------------------------------------------
# 4. GRÁFICO INTERACTIVO (Plotly)
# ------------------------------------------------------------------------------
# [cite_start]Requisito explícito de la práctica [cite: 663]
p_inter <- ggplot(df_analysis, aes(x = age, y = fare_log, color = survived, 
                                   text = paste("Clase:", pclass, 
                                                "<br>Sexo:", sex,
                                                "<br>Familia:", family_size))) +
  geom_point(alpha = 0.6) +
  labs(title = "Exploración Interactiva: Edad vs Tarifa", x = "Edad", y = "Log(Tarifa)") +
  theme_minimal()

ggplotly(p_inter, tooltip = c("x", "y", "color", "text"))