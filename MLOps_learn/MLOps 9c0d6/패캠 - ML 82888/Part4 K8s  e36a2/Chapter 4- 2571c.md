# Chapter 4-1. Jenkins 기본 및 Install

### Jenkins for MLOps

![Untitled](Chapter%204-%202571c/Untitled.png)

- 커밋할 때 감지하여 서버에 자동으로 반영시켜줌
    - 기존에 작업들이 정의되어 있어 이를 통해 test, 특정 조건에서만 서버에 반영 등 을 수행할 수 있음.
    

 

![Untitled](Chapter%204-%202571c/Untitled%201.png)

Jenkins 특징

- Jenkinsfile
    - `Jenkinsfile` 을 이용해 **Job 혹은 파이프라인을 정의**할 수 있다. `Jenkinsfile` 덕분에 일반 소스코드를 다루는 Github 업로드, Vscode 로 수정하는 것으로 파일을 이용할 수 있음
    - 기본적으로 `Jenkinsfile`을 통해 젠킨스를 실행함
- Scripted Pipeline (스크립트 파이프라인)
    - Jenkins 관련 구조를 자세히 가지지 않고 프로그램의 흐름을 Java 와 유사한 [Groovy](https://groovy-lang.org/) 라는 동적 객체 지향 프로그래밍 언어를 이용해 관리되었음
    - 매우 유연하지만 시작하기가 어려움
    
    ```groovy
    node { ## 빌드를 수행할 node 또는 agent를 의미한다.
        stage("Stage 1"){
            echo "Hello"
        }
        stage("Stage 2"){
            echo "World"
            sh "sleep 5"
        }
        stage("Stage 3"){
            echo "Good to see you!"
        }
    }
    ```
    
    ![Untitled](Chapter%204-%202571c/Untitled%202.png)
    
- Declarative Pipeline (선언적 파이프라인)
    - 2016년 경 [Cloudbees 에서 개발](https://docs.cloudbees.com/docs/admin-resources/latest/pipeline-syntax-reference-guide/declarative-pipeline)
    - **사전에 정의된 구조만** 사용할 수 있기 때문에 **CI/CD 파이프라인이 단순한 경우에 적합**하며 아직은 많은 제약사항이 따른다.
    - [공식 문서](https://www.jenkins.io/doc/book/pipeline/syntax/)
    
    ```bash
    pipeline {
        agent any
        stages {
            stage('Stage 1') {
                steps {
                    script {
                        echo 'Hello'
                    }
                }
            }
    
            stage('Stage 2') {
                steps {
                    script {
                        echo 'World'
                        sh 'sleep 5'
                    }
                }
            }
    
            stage('Stage 3') {
                steps {
                    script {
                        echo 'Good to see you!'
                    }
                }
            }
        }
    }
    ```
    
    ![Untitled](Chapter%204-%202571c/Untitled%203.png)
    
    즉 `jenkinsfile`을 미리 받아서 그곳에 작업을 선언해두어야 함.
    
    ---
    
    ### **Jenkins 설치**
    
    - `Local` 에 직접 설치해보기
        - JDK 설치
            
            ```bash
            sudo apt install openjdk-11-jre-headless
            ```
            
        - Key 다운로드
            
            ```bash
            wget -q -O - https://pkg.jenkins.io/debian/jenkins.io.key | sudo apt-key add -
            echo deb http://pkg.jenkins.io/debian-stable binary/ | sudo tee /etc/apt/sources.list.d/jenkins.list
            ```
            
        - Jenkins 설치하기
            
            ```bash
            # sudo apt-get update
            sudo apt-get install jenkins
            ```
            
        - 정상여부 확인
            
            ```bash
            sudo systemctl status jenkins
            # 재시작 : sudo service jenkins restart
            ```
            
        - 초기 패스워드 확인
            
            ```bash
            sudo cat /var/lib/jenkins/secrets/initialAdminPassword
            ```
            
        - 브라우저 접속 (localhost:8080)
        → `**172.17.0.1:8099**`
            - 초기 비밀번호 사용하여 로그인
        - 플러그인 설치
            
            ![Untitled](Chapter%204-%202571c/Untitled%204.png)
            
        - 계정 만들기
            - admin / 1234
            
            ![Untitled](Chapter%204-%202571c/Untitled%205.png)
            
            ---
            
            # Practice
            
            ### **Jenkinsfile 의 기본적인 구조를 알아보고 생성한다.**
            
            - 기본 코드 구조
                - pipeline : 반드시 맨 위에 있어야 한다.
                - agent : 어디에서 실행할 것인지 정의한다.
                    - any, none, label, node, docker, dockerfile, kubernetes
                    - agent 가 none 인 경우 stage 에 포함시켜야 함
                        
                        ```bash
                        pipeline {
                            agent none 
                            stages {
                                stage('Example Build') {
                                    agent { docker 'maven:3-alpine' }
                                    steps {
                                        echo 'Hello, Maven'
                                        sh 'mvn --version'
                                    }
                                }
                                stage('Example Test') {
                                    agent { docker 'openjdk:8-jre' }
                                    steps {
                                        echo 'Hello, JDK'
                                        sh 'java -version'
                                    }
                                }
                            }
                        }
                        ```
                        
                - stages : 하나 이상의 stage 에 대한 모음
                    - pipeline 블록 안에서 한 번만 실행 가능함
                        
                        ```yaml
                        pipeline {
                        	agent any
                        	stages {
                        		stage("build") {
                        			steps {
                        				echo 'building the applicaiton...'
                        			}
                        		}
                        		stage("test") {
                        			steps {
                        				echo 'testing the applicaiton...'
                        			}
                        		}
                        		stage("deploy") {
                        			steps {
                        				echo 'deploying the applicaiton...'
                        			}
                        		}
                        	}
                        }
                        ```
                        
            - 본인 Github 에 새 Repository [ js-pipeline-project ] 생성
            - Local 에서 'Jenkinsfile' 파일 생성하여 위 stages 코드 복사한 후 Github 업로드
                
                ![Untitled](Chapter%204-%202571c/Untitled%206.png)
                
                - 위와 같이 일단 Repo 생성
                
                ![Untitled](Chapter%204-%202571c/Untitled%207.png)
                
                - `jenkinsfile` 생성 후 깃커밋
            
            ### **Jenkins 에서 Pipeline Job 을 생성하고 빌드한다.**
            
            - Pipeline 생성
                - 이름은 'jenkins-pipeline' 으로 입력
            
            ![Untitled](Chapter%204-%202571c/Untitled%208.png)
            
            - `Pipeline` 으로 생성
            
            ![Untitled](Chapter%204-%202571c/Untitled%209.png)
            
            ![Untitled](Chapter%204-%202571c/Untitled%2010.png)
            
            - 이때 `branch` 는 `main` 으로
            
            ![Untitled](Chapter%204-%202571c/Untitled%2011.png)
            
            - 이때 `Jenkinsfile` 파일명을 맨앞에 대문자로 써주어야 함.
            → 안그러면 못 찾고 에러남..
            
            ![Untitled](Chapter%204-%202571c/Untitled%2012.png)
            
            - 로그도 볼 수 있음
            
            - Git 추가
                - [General]-[Branch Sources]-[Add source 선택]-[Git]
                - 위에서 생성한 Github Repository 추가
                - Credentials - [Add]
                - [Save] - Log 출력
            - Pipeline Status 확인
                - 각 stage log 확인
            
            ### **post 를 이용해 모든 stage 가 실행된 후의 명령을 정의한다.**
            
            - post 조건
                - always, changed, fixed, regression, aborted, success, unsuccessful, unstable, failure, notBuilt, cleanup
                
                ```yaml
                pipeline {
                	agent any
                	stages {
                		stage("build") {
                			steps {
                				echo 'building the applicaiton...'
                			}
                		}
                		stage("test") {
                			steps {
                				echo 'testing the applicaiton...'
                			}
                		}
                		stage("deploy") {
                			steps {
                				echo 'deploying the applicaiton...'
                			}
                		}
                	}
                	post {
                			always {
                				echo 'building..'
                			}
                			success {
                	            echo 'success'
                			}
                			failure {
                	            echo 'failure'
                			}
                		}
                	}
                ```
                
            
            ![Untitled](Chapter%204-%202571c/Untitled%2013.png)
            
            위와 같이 중간에 `env.GIT_BRANCH` 를 추가해주면 branch 현황도 알 수 있음
            
            ![Untitled](Chapter%204-%202571c/Untitled%2014.png)
            
            - 로그에 브랜치 위치 출력
            - 
            
            ### **when 을 이용해 stage 가 실행되는 조건을 추가 해본다.**
            
            - when 조건 추가
                - test stage 에서 Branch 이름에 따른 조건 추가
                - build stage 에서 Branch 이름에 따른 조건 추가
                
                ```bash
                pipeline {
                	agent any
                	stages {
                		stage("build") {
                			when {
                				expression {
                					env.GIT_BRANCH == 'origin/main'
                				}
                			}
                			steps {
                				echo 'building the applicaiton...'
                			}
                		}
                		stage("test") {
                			when {
                				expression {
                					env.GIT_BRANCH == 'origin/test' || env.GIT_BRANCH == ''
                				}
                			}
                			steps {
                				echo 'testing the applicaiton...'
                			}
                		}
                		stage("deploy") {
                			steps {
                				echo 'deploying the applicaiton...'
                			}
                		}
                	}
                }
                ```
                
            - 해당 내용은 test는 branch 가 test 일 경우에만 실행해야하는데 그렇지 않으므로 패스
            
            ![Untitled](Chapter%204-%202571c/Untitled%2015.png)
            
            - 이런 결과가 나옴
            
            ### **Jenkinsfile 환경변수를 설정 해본다.**
            
            - Jenkinsfile 자체 환경변수 목록 보기
                - [http://localhost:8080/env-vars.html/](http://localhost:8080/env-vars.html/) 접속 / 또는 [Jenkins pipeline-syntax](https://opensource.triology.de/jenkins/pipeline-syntax/globals) 참고
            - Custom 환경변수 사용하기
                - echo 사용 시 큰 따옴표 주의
                
                ```yaml
                pipeline {
                	agent any
                	environment {
                		NEW_VERSION = '1.0.0'
                	}
                	stages {
                		stage("build") {
                			steps {
                				echo 'building the applicaiton...'
                				echo "building version ${NEW_VERSION}"
                			}
                		}
                		stage("test") {
                			steps {
                				echo 'testing the applicaiton...'
                			}
                		}
                		stage("deploy") {
                			steps {
                				echo 'deploying the applicaiton...'
                			}
                		}
                	}
                }
                ```
                
            
            ![Untitled](Chapter%204-%202571c/Untitled%2016.png)
            
            - Credentials 자격 증명 환경 변수로 사용하기
                - Jenkins credential 추가
                    - [Jenkins 관리]-[Manage Credentials]-[Global credentials]-[Add credentials]
                    
                    ![Untitled](Chapter%204-%202571c/Untitled%2017.png)
                    
                    ![Untitled](Chapter%204-%202571c/Untitled%2018.png)
                    
                    ![Untitled](Chapter%204-%202571c/Untitled%2019.png)
                    
                    - Username : admin / Password : 1234 / ID : admin
                - Jenkinsfile 에서 환경변수로 사용
                    
                    ```yaml
                    pipeline {
                    	agent any
                    	environment {
                    		NEW_VERSION = '1.0.0'
                    		ADMIN_CREDENTIALS = credentials('admin_user_credentials')
                    	}
                    	stages {
                    		stage("build") {
                    			steps {
                    				echo 'building the applicaiton...'
                    				echo "building version ${NEW_VERSION}"
                    			}
                    		}
                    		stage("test") {
                    			steps {
                    				echo 'testing the applicaiton...'
                    			}
                    		}
                    		stage("deploy") {
                    			steps {
                    				echo 'deploying the applicaiton...'
                    				echo "deploying with ${ADMIN_CREDENTIALS}"
                    				sh 'printf ${ADMIN_CREDENTIALS}'
                    			}
                    		}
                    	}
                    }
                    ```
                    
                    - 해당 환경변수 인 경우에만 실행
                    
                    ![Untitled](Chapter%204-%202571c/Untitled%2020.png)
                    
                - Jenkins 플러그인 중 Credentials Plugin 확인
                    
                    ![Untitled](Chapter%204-%202571c/Untitled%2021.png)
                    
                    ![Untitled](Chapter%204-%202571c/Untitled%2022.png)
                    
                    ![Untitled](Chapter%204-%202571c/Untitled%2023.png)
                    
                    ```yaml
                    pipeline {
                    	agent any
                    	environment {
                    		NEW_VERSION = '1.0.0'
                    	}
                    	stages {
                    		stage("build") {
                    			steps {
                    				echo 'building the applicaiton...'
                    				echo "building version ${NEW_VERSION}"
                    			}
                    		}
                    		stage("test") {
                    			steps {
                    				echo 'testing the applicaiton...'
                    			}
                    		}
                    		stage("deploy") {
                    			steps {
                    				echo 'deploying the applicaiton...'
                    				withCredentials([[$class: 'UsernamePasswordMultiBinding',
                    					credentialsId: 'admin', 
                    					usernameVariable: 'USER', 
                    					passwordVariable: 'PWD'
                    				]]) {
                    					sh 'printf ${USER}'
                    				}
                    			}
                    		}
                    	}
                    }
                    ```
                    
            
            ### **parameters 를 이용하는 방법을 알아본다.**
            
            - Jenkinsfile 에 parameter 추가
                
                ```yaml
                pipeline {
                	agent any
                	parameters {
                		string(name: 'VERSION', defaultValue: '', description: 'deployment version')
                		choice(name: 'VERSION', choices: ['1.1.0','1.2.0','1.3.0'], description: '')
                		booleanParam(name: 'executeTests', defaultValue: true, description: '')
                	}
                	stages {
                		stage("build") {
                			steps {
                				echo 'building the applicaiton...'
                			}
                		}
                		stage("test") {
                			steps {
                				echo 'testing the applicaiton...'
                			}
                		}
                		stage("deploy") {
                			steps {
                				echo 'deploying the applicaiton...'
                			}
                		}
                	}
                }
                ```
                
            
            ![Untitled](Chapter%204-%202571c/Untitled%2024.png)
            
            - 다음과 같이 왼쪽 칸에도 `build with parameter` 라고 생성되고 누르면 위와 같이 변경되어 나옴
            
            - executeTests 가 true 인 경우의 조건 추가해보기
                
                ```yaml
                pipeline {
                	agent any
                	parameters {
                		choice(name: 'VERSION', choices: ['1.1.0','1.2.0','1.3.0'], description: '')
                		booleanParam(name: 'executeTests', defaultValue: true, description: '')
                	}
                	stages {
                		stage("build") {
                			steps {
                				echo 'building the applicaiton...'
                			}
                		}
                		stage("test") {
                			when {
                				expression {
                					params.executeTests
                				}
                			}
                			steps {
                				echo 'testing the applicaiton...'
                			}
                		}
                		stage("deploy") {
                			steps {
                				echo 'deploying the applicaiton...'
                				echo "deploying version ${params.VERSION}"
                			}
                		}
                	}
                }
                ```
                
            
            ![Untitled](Chapter%204-%202571c/Untitled%2025.png)
            
            - 체크를 풀면 `test`는 넘어가고 실행한다
            
            - 실제 Jenkinsfile 에 적용해 본다.
                - Build with Parameters 실행
                    - 1.2.0 버전 선택
                    - executeTests 선택 해제
                - test stage 를 건너뛰고 실행되는지 확인
            
            ### **외부 groovy scripts 를 만들어 사용해 본다.**
            
            - groovy script 추가
                
                [script.groovy]
                
                ```groovy
                
                def buildApp() {
                	echo 'building the applications...'
                }
                
                def testApp() {
                	echo 'testing the applications...'
                }
                
                def deployApp() {
                	echo 'deploying the applicaiton...'
                	echo "deploying version ${params.VERSION}"
                }
                return this
                ```
                
                [Jenkinsfile]
                
                ```yaml
                pipeline {
                	agent any
                	parameters {
                		choice(name: 'VERSION', choices: ['1.1.0','1.2.0','1.3.0'], description: '')
                		booleanParam(name: 'executeTests', defaultValue: true, description: '')
                	}
                	stages {
                		stage("init") {
                			steps {
                				script {
                					gv = load "script.groovy"
                				}
                			}
                		}
                		stage("build") {
                			steps {
                				script {
                					gv.buildApp()
                				}
                			}
                		}
                		stage("test") {
                			when {
                				expression {
                					params.executeTests
                				}
                			}
                			steps {
                				script {
                					gv.testApp()
                				}
                			}
                		}
                		stage("deploy") {
                			steps {
                				script {
                					gv.deployApp()
                				}
                			}
                		}
                	}
                }
                ```
                
                - Jenkinsfile 의 모든 환경변수는 groovy script 에서 사용 가능하다.
                - Github Repo 에 반영하고 실행/로그 확인
                - 빌드 결과 확인
            
            ![Untitled](Chapter%204-%202571c/Untitled%2026.png)
            
            ### **Replay 에서 수정 후 빌드 다시 실행해 보기**
            
            - testApp 에 echo 'Replay' 를 추가 후 다시 빌드
                
                ![Untitled](Chapter%204-%202571c/Untitled%2027.png)
                
            
            ![Untitled](Chapter%204-%202571c/Untitled%2028.png)