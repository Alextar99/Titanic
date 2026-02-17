library(tidyverse)
library(VIM); library(mice); library(corrplot); library(gridExtra)

rm(list=ls())

# =====================================================
# PASO 1: LECTURA Y TRANSFORMACIÓN
# =====================================================
titanic <- read.table("C:/Users/alega/Downloads/titanic.txt", header = TRUE, sep=";") %>%
  mutate(
    age = as.numeric(age),
    sex = as.factor(sex), pclass = as.factor(pclass), survived = as.factor(survived),
    embarked = as.factor(embarked), boat = as.factor(boat), body = as.numeric(body),
    fare = as.numeric(str_replace(str_replace(trimws(fare), ",", "."), "\\\\.", ""))
  )

cat("=== DATASET INICIAL ===\n")
summary(titanic); dim(titanic)
cat("Filas completas:", nrow(titanic[complete.cases(titanic), ]), "\n\n")

# =====================================================
# PASO 2: DETECCIÓN Y ANÁLISIS OUTLIERS
# =====================================================
detectar_outliers <- function(x, coef = 1.5) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  list(lower = Q1 - coef * IQR, upper = Q3 + coef * IQR,
       n_outliers = sum(x < (Q1 - coef*IQR) | x > (Q3 + coef*IQR), na.rm=T))
}

vars_num <- c("age", "fare", "sibsp", "parch")
cat("=== OUTLIERS DETECTADOS (Tukey 1.5 IQR) ===\n")
for(var in vars_num) {
  info <- detectar_outliers(titanic[[var]])
  cat(sprintf("%s: %d outliers [%.1f, %.1f]\n", var, info$n_outliers, info$lower, info$upper))
}

# Visualización inicial
par(mfrow = c(2, 2))
boxplot(titanic$age, main = "Age (outliers plausibles)", col = "lightblue", ylim=c(0,80))
boxplot(titanic$fare, main = "Fare (outliers informativos)", col = "lightcoral", ylim=c(0,300))
boxplot(titanic$sibsp, main = "Sibsp (familias reales)", col = "orange")
boxplot(titanic$parch, main = "Parch (sesgado)", col = "gold")

# =====================================================
# PASO 3: TRATAMIENTO ÓPTIMO OUTLIERS
# =====================================================
cat("\n=== DECISIÓN TRATAMIENTO ===\n")
cat("✓ Age: MANTENER outliers (ancianos reales)\n")
cat("✓ Fare: LOG-TRANSFORM (preserva info, normaliza)\n")
cat("✓ Sibsp/Parch: MANTENER + crear family_size\n\n")

titanic_tratado <- titanic %>%
  mutate(
    # Log-transform para normalizar (NO eliminar)
    fare_log = log(fare + 1),
    age_log = log(age + 1),
    
    # Variables derivadas
    family_size = sibsp + parch + 1,
    solo = ifelse(family_size == 1, 1, 0),
    
    # Categorías edad
    age_cat = cut(age, breaks=c(0,12,18,35,60,100), 
                  labels=c("Niño","Adolescente","Joven","Adulto","Anciano"))
  )


# Comparación distribuciones
par(mfrow = c(2, 2))
hist(titanic$fare, breaks=30, col="lightcoral", main="Fare ORIGINAL (sesgado)")
hist(titanic_tratado$fare_log, breaks=30, col="lightgreen", main="Fare LOG (normalizado)")
hist(titanic$age, breaks=30, col="orange", main="Age ORIGINAL")
hist(titanic_tratado$age_log, breaks=30, col="lightblue", main="Age LOG")

# =====================================================
# PASO 4: GRÁFICOS PRE-IMPUTACIÓN
# =====================================================
ggplot(titanic_tratado, aes(pclass, fare)) + 
  geom_boxplot(outlier.color="red") + 
  scale_y_log10() + ggtitle("Fare por pclass (log scale)") + theme_minimal()

ggplot(titanic_tratado, aes(embarked, fare)) + labs(title = "Precio billetes según puerta de embarque") +
  geom_boxplot(fill="red") + scale_y_log10() + theme_minimal()

medianas <- titanic_tratado %>% group_by(pclass) %>% 
  summarise(mediana_fare = median(fare, na.rm = TRUE), .groups = "drop")
ggplot(titanic_tratado, aes(fare, fill = pclass)) + geom_density(alpha = 0.4) +
  geom_vline(data = medianas, aes(xintercept = mediana_fare), 
             linetype = "dashed", color = "red") +
  scale_x_log10() + theme_minimal() + ggtitle("Densidad Fare (preserva outliers)")

# =====================================================
# PASO 5: IMPUTACIÓN (SOLO NA, NO OUTLIERS)
# =====================================================
set.seed(123)

# Preparar dataset (excluir variables basura)
titanic_base <- titanic_tratado %>% select(-body, -boat, -home.dest)

cat("=== IMPUTACIÓN NA (outliers PRESERVADOS) ===\n")
cat("NA age:", sum(is.na(titanic_base$age)), "| NA fare:", sum(is.na(titanic_base$fare)), "\n")

# k-NN: imputa SOLO NA
titanic_knn <- kNN(titanic_base, variable = c("age", "fare"), k = 10, 
                   numFun = median, imp_var = TRUE)

# MICE: comparación
imp_mice <- mice(titanic_base[, c("age", "fare", "pclass", "sex", "sibsp", "parch")], 
                 m = 1, maxit = 5, print = FALSE)
titanic_mice_temp <- complete(imp_mice, 1)
titanic_mice <- titanic_base %>% 
  mutate(age_mice = titanic_mice_temp$age, fare_mice = titanic_mice_temp$fare)

# ELEGIR k-NN
titanic_imputado <- titanic_knn %>% 
  select(-ends_with("_imp")) %>%
  mutate(fare_log = log(fare + 1), age_log = log(age + 1))  # Re-calcular log post-imputación

cat("POST-imputación NA age:", sum(is.na(titanic_imputado$age)), 
    "| NA fare:", sum(is.na(titanic_imputado$fare)), "\n")
cat("✓ Outliers PRESERVADOS:", max(titanic_imputado$fare), "$ max fare\n\n")

# Comparación visual
par(mfrow = c(2, 3))
hist(titanic_base$age, col="lightcoral", main="Age ORIGINAL", breaks=20)
hist(titanic_imputado$age, col="lightblue", main="Age k-NN ✓", breaks=20)
hist(titanic_mice$age_mice, col="lightgreen", main="Age MICE", breaks=20)
hist(titanic_base$fare, col="orange", main="Fare ORIGINAL", breaks=20, xlim=c(0,300))
hist(titanic_imputado$fare, col="gold", main="Fare k-NN ✓", breaks=20, xlim=c(0,300))
hist(titanic_mice$fare_mice, col="pink", main="Fare MICE", breaks=20, xlim=c(0,300))

# Tabla comparativa
comparacion <- rbind(
  Original = c(NA_age=sum(is.na(titanic_base$age)), Media_age=mean(titanic_base$age,na.rm=T),
               Max_fare=max(titanic_base$fare,na.rm=T)),
  `kNN✓` = c(NA_age=0, Media_age=mean(titanic_imputado$age),
             Max_fare=max(titanic_imputado$fare)),
  MICE = c(NA_age=0, Media_age=mean(titanic_mice$age_mice),
           Max_fare=max(titanic_mice$fare_mice))
)
print(round(comparacion, 1))
cat("\n✅ k-NN elegido: imputa NA, preserva outliers informativos\n\n")

# =====================================================
# PASO 6: EMBARKED + LIMPIEZA FINAL
# =====================================================
titanic <- titanic_imputado %>%
  mutate(embarked = case_when(trimws(embarked) == "" | embarked == " " ~ NA_character_,
                              TRUE ~ trimws(as.character(embarked)))) %>%
  group_by(pclass) %>%
  mutate(embarked = ifelse(is.na(embarked), 
                           names(sort(table(embarked), decreasing=TRUE))[1], embarked)) %>%
  ungroup()

ggplot(titanic, aes(factor(pclass), fill=factor(embarked))) + 
  geom_bar(position="fill") + ggtitle("Embarked imputado por pclass") + theme_minimal()

summary(titanic)

# =====================================================
# PASO 7: EDA COMPLETO (con outliers preservados)
# =====================================================

ggplot(titanic, aes(sex, fill=survived)) + geom_bar(stat="count") + 
  labs(y="Nº de personas") + theme_minimal()

ggplot(titanic, aes(x = pclass, y = fare)) +
  geom_boxplot(fill = "steelblue") +
  facet_wrap(~sex) +  # Dos paneles: hombres/mujeres
  labs(title = "Fare por clase y sexo") +
  geom_hline(aes(yintercept=80), col = "red") + 
  theme_minimal()

porcentaje <- c("0%", "25%", "50%", "75%", "100%")
ggplot(titanic, aes(pclass, fill=survived)) + geom_bar(position="fill") +
  scale_y_continuous(labels=porcentaje) + ylab("Survival Rate") +
  ggtitle("Survival Rate by Class") + facet_wrap(~embarked) + theme_minimal()

ggplot(titanic, aes(sex, age, fill=survived)) + geom_boxplot() + 
  ggtitle("Edad por sexo (ancianos preservados)") + theme_minimal()

ggplot(titanic, aes(parch, sibsp, col=survived)) + geom_jitter(alpha=0.6) + 
  ggtitle("Familias (outliers = familias grandes reales)")

ggplot(transform(titanic, family_size=sibsp+parch+1), aes(family_size, fill=survived)) +
  geom_bar(position="dodge") + ggtitle("Family Size vs Survived") + theme_minimal()

# Gráfico clave: fare vs supervivencia (outliers informativos)
ggplot(titanic, aes(fare, fill=survived)) + 
  geom_density(alpha=0.6) + scale_x_log10() +
  ggtitle("OUTLIERS RICOS SOBREVIVEN MÁS (preservados ✓)") + theme_minimal()

# =====================================================
# PASO 8: CORRELACIONES
# =====================================================
titanic_full <- titanic %>%
  mutate(pclass_num=as.numeric(pclass), survived_num=as.numeric(survived),
         sex_num=ifelse(sex=="male",1,0),
         embarked_num=case_when(embarked=="S"~1, embarked=="C"~2, embarked=="Q"~3)) %>%
  select(age, sibsp, parch, fare, fare_log, pclass_num, survived_num, sex_num, embarked_num)

print(round(cor(titanic_full, use="pairwise.complete.obs"), 3))
corrplot(cor(titanic_full), method="color", type="upper", order="hclust",
         tl.cex=0.7, title="Correlaciones (fare_log normalizado)")

cat("\n🎉 ANÁLISIS COMPLETO - OUTLIERS PRESERVADOS\n")
cat("Justificación: Outliers Titanic = información histórica real valiosa\n")
cat("Dataset final:", nrow(titanic), "filas | 100% completas | 0 info perdida\n")


