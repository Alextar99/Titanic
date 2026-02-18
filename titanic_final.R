
# ==============================================================================
# CARGA DE LIBRERÍAS
# ==============================================================================

if (!require("pacman")) install.packages("pacman")
library(pacman)

# Cargamos librerías de Práctica 1 y 2
p_load(tidyverse, VIM, moments, plotly, fastDummies, corrplot, scales)

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


# Feature engineering: family size
df_clean$family_size <- df_clean$sibsp + df_clean$parch + 1




# 4. Tratamiento de Outliers (Fare)
# ------------------------------------------------------------------------------

# Detección visual
boxplot(df_clean$fare, main = "Outliers en Fare (Antes)")

# Transformación Logarítmica para reducir asimetría 
df_clean$fare_log <- log(df_clean$fare + 1)  # +1 para evitar log(0)

# Comprobación de asimetría
cat("Fare Original:", skewness(df_clean$fare), "\n")
cat("Fare Log:", skewness(df_clean$fare_log), " (Más cercano a 0 es mejor)\n")

boxplot(df_clean$fare_log, main = "Outliers en Fare (Después)")




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
# B) ANÁLISIS EXPLORATORIO DE DATOS (EDA) - VERSIÓN FINAL
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. ANÁLISIS UNIVARIANTE
# ------------------------------------------------------------------------------


# --- Variable Numérica: EDAD (Histograma + Densidad) ---

p1 <- ggplot(df_analysis, aes(x = age)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "white") +
  geom_density(alpha = 0.2, fill = "red") +
  theme_minimal() +
  labs(title = "Distribución de la Edad", 
       subtitle = "La imputación k-NN preserva la distribución original",
       x = "Edad (Años)", y = "Densidad")
print(p1)



# --- Test de normalidad ---

# Shapiro-Wilk
test_shapiro <- shapiro.test(df_analysis$age)
print(test_shapiro)


# Interpretación:
# p < 0.05 ---> la distribución no sigue una distribución normal

if(test_shapiro$p.value < 0.05) {
  cat("INTERPRETACIÓN: P-valor < 0.05.\n")
  cat("-> RECHAZAMOS la normalidad. La edad NO sigue una distribución normal.\n")
  cat("-> Por tanto, la MEDIANA es mejor medida de centralidad que la media.\n")
} else {
  cat("INTERPRETACIÓN: P-valor > 0.05.\n")
  cat("-> No podemos rechazar la normalidad (los datos parecen normales).\n")
}

qqnorm(df_analysis$age, main = "QQ-Plot de la Edad", pch = 19, col = "gray50"); qqline(df_analysis$age, col = "red", lwd = 2)






# --- Variable Numérica: TARIFA (Validación de Transformación) ---

p2 <- ggplot(df_analysis, aes(x = fare_log)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribución de la Tarifa (Log)", 
       subtitle = "La transformación logarítmica ha corregido el sesgo positivo",
       x = "Log(Tarifa)", y = "Frecuencia")
print(p2)



# --- Variable Categórica: SUPERVIVENCIA ---

p3 <- ggplot(df_analysis, aes(x = survived, fill = survived)) +
  geom_bar() +
  scale_fill_manual(values = c("firebrick", "forestgreen")) +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  theme_minimal() +
  labs(title = "Tasas de Supervivencia", x = "¿Sobrevivió?", y = "Pasajeros")
print(p3)





# ------------------------------------------------------------------------------
# 2. ANÁLISIS BIVARIANTE (Relaciones Clave)
# ------------------------------------------------------------------------------


# --- CLASE vs SUPERVIVENCIA ---

# A mejor clase, mayor probabilidad de sobrevivir
p4 <- ggplot(df_analysis, aes(x = pclass, fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  labs(title = "Probabilidad de Supervivencia por Clase",
       x = "Clase", y = "Porcentaje")
print(p4)





# --- PUERTO DE EMBARQUE vs SUPERVIVENCIA ---
# Hipótesis: El puerto influye indirectamente debido a la clase social de los pasajeros que embarcaron.

# 1. Gráfico de Barras Apiladas (Porcentajes)
p_embarked <- ggplot(df_analysis, aes(x = embarked, fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Set1") + # Mantenemos la paleta del resto del script
  theme_minimal() +
  labs(title = "Supervivencia según Puerto de Embarque",
       subtitle = "Cherbourg (C) presenta la mayor tasa de supervivencia (más 1ra clase)",
       x = "Puerto (C = Cherbourg, Q = Queenstown, S = Southampton)", 
       y = "Porcentaje")
print(p_embarked)

# 2. Validación Estadística (Chi-Cuadrado)
# H0: La supervivencia es independiente del puerto.
# H1: Hay relación entre el puerto y sobrevivir.
test_embarked <- chisq.test(table(df_analysis$embarked, df_analysis$survived))
print(test_embarked)

# Interpretación automática
if(test_embarked$p.value < 0.05) {
  cat("RESULTADO: p-value < 0.05. Hay diferencias significativas según el puerto.\n")
} else {
  cat("RESULTADO: No hay diferencias significativas.\n")
}


# --- VALIDACIÓN DE HIPÓTESIS: CLASE vs PUERTO ---

# Gráfico de Barras Apiladas (Clase dentro de cada Puerto)
p_embarked_class <- ggplot(df_analysis, aes(x = embarked, fill = pclass)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Oranges") + # Usamos otra paleta para distinguir
  theme_minimal() +
  labs(title = "Distribución de Clases por Puerto",
       x = "Puerto de Embarque", 
       y = "Porcentaje de Pasajeros")
print(p_embarked_class)

# Tabla de Contingencia (Para ver los números exactos)
tabla_puerto_clase <- table(df_analysis$embarked, df_analysis$pclass)
print(addmargins(tabla_puerto_clase))

# Porcentajes por fila (Para ver qué % de C eran de 1ra)
print(round(prop.table(tabla_puerto_clase, margin = 1) * 100, 1))


# El análisis visual y el test Chi-Cuadrado muestran que en Cherbourg la tasa de supervivencia es más alta, pero esto se debe 
# a que en ese puerto embarcó una proporción mucho mayor de pasajeros de Primera Clase en comparación con Queenstown, 
# donde la mayoría eran emigrantes de Tercera Clase. Por tanto, el puerto actúa como una variable 'proxy' del nivel socioeconómico.






# --- EDAD vs SUPERVIVENCIA (Violin Plot + Boxplot) ---

p5 <- ggplot(df_analysis, aes(x = survived, y = age, fill = survived)) +
  geom_violin(trim = FALSE, alpha = 0.5) +          # Muestra la densidad (forma)
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) + # Muestra la mediana
  theme_minimal() +
  labs(title = "Distribución de Edad según Supervivencia", 
       subtitle = "La forma de 'violín' en Sí muestra mayor densidad de niños",
       x = "¿Sobrevivió?", y = "Edad")
print(p5)



# --- TAMAÑO FAMILIA vs SUPERVIVENCIA ---

p6 <- ggplot(df_analysis, aes(x = as.factor(family_size), fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  labs(title = "Supervivencia según Tamaño Familiar", 
       subtitle = "Las familias pequeñas (2-4) tuvieron mayor tasa de supervivencia",
       x = "Miembros de la Familia", y = "Porcentaje")
print(p6)




# ------------------------------------------------------------------------------
# 3. ANÁLISIS MULTIVARIANTE
# ------------------------------------------------------------------------------


#  Patrón 'Mujeres y niños primero'
p7 <- ggplot(df_analysis, aes(x = pclass, fill = survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~sex) + 
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("gray40", "dodgerblue")) +
  theme_minimal() +
  labs(title = "Supervivencia por Clase y Sexo",
       x = "Clase", y = "Porcentaje")
print(p7)



# --- MATRIZ DE CORRELACIÓN ---

# Preparamos datos numéricos
df_cor <- df_analysis %>%
  mutate(sex_num = as.numeric(sex), 
         survived_num = as.numeric(survived) - 1) %>%
  select(age, fare_log, family_size, sex_num, survived_num)

M <- cor(df_cor, use = "pairwise.complete.obs")

# Mapa de calor
corrplot(M, method = "color", type = "upper", order = "hclust", 
         addCoef.col = "black", tl.col = "black", diag = FALSE,
         title = "Mapa de Calor de Correlaciones", mar = c(0,0,1,0))






# ------------------------------------------------------------------------------
# 4. GRÁFICO INTERACTIVO
# ------------------------------------------------------------------------------


# Definimos el gráfico
p_inter <- ggplot(df_analysis, aes(x = pclass, y = age, color = survived)) +
  
  # 1. Caja de fondo (Transparente) para referencia estadística
  geom_boxplot(width = 0.4, fill = "white", alpha = 0, outlier.shape = NA) +
  
  # 2. Puntos con Jitter
  geom_jitter(aes(text = paste("<b>Pasajero</b>",
                               "<br>Edad:", age,
                               "<br>Sexo:", sex,
                               "<br>Clase:", pclass,
                               "<br>Tarifa: $", round(fare, 2),
                               "<br>Familia:", family_size)),
              width = 0.2, 
              alpha = 0.6, 
              size = 2) +
  
  # Estética
  scale_color_manual(values = c("firebrick", "forestgreen"), name = "Sobrevivió") +
  labs(title = "Distribución Interactiva: Edad por Clase", 
       subtitle = "Pasa el ratón para ver detalles. (Verde = Sobrevivió)",
       x = "Clase", y = "Edad") +
  theme_minimal()

# Generar el gráfico interactivo
ggplotly(p_inter, tooltip = "text")
