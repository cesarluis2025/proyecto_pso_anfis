pipeline {
    agent any

    stages {

        stage('Clonar repositorio') {
            steps {
                echo 'Clonando el repositorio desde GitHub...'
                checkout scm
            }
        }

        stage('Construir imagen Docker') {
            steps {
                echo 'Construyendo la imagen Docker del proyecto...'
                bat 'docker-compose build'
            }
        }

        stage('Ejecutar pipeline PSO-ANFIS') {
            steps {
                echo 'Ejecutando el modelo PSO-ANFIS dentro del contenedor...'
                bat 'docker-compose run --rm jupyter python main.py'
            }
        }

        stage('Verificar resultados') {
            steps {
                echo 'Verificando que se generaron los archivos de resultados...'
                bat 'if exist results\\tables\\comparativa.csv (echo Tabla generada correctamente) else (exit 1)'
                bat 'if exist results\\figures\\fig4_convergencia_pso.png (echo Graficas generadas correctamente) else (exit 1)'
            }
        }

        stage('Limpiar contenedor') {
            steps {
                echo 'Limpiando contenedores Docker...'
                bat 'docker-compose down'
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado exitosamente.'
        }
        failure {
            echo 'El pipeline fallo. Revisar los logs.'
        }
    }
}
