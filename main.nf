if (nextflow.version.matches(">=20.07.1")){
    nextflow.enable.dsl=2
}else{
    // Support lower version of nextflow
    nextflow.preview.dsl=2
}

// set utils dirs
projectDir = workflow.projectDir


params.error = "ERROR!"
params.result = "Predicting......"


// Check all tools work well
process envCheck {
    tag "envcheck"
    errorStrategy 'terminate'

    label 'process_low'

    output:
    path 'error.txt', emit: error
    
    """
    python ${projectDir}/check_data.py \
        --log-path ${projectDir}/wrong.log \
        --input-dir ${projectDir}/test_data \
        --output-path ${projectDir}/test.json
    echo ${params.error} > error.txt
    echo "Check data done"
    """
}

process printLog{
    tag "printWrongInfo"

    label 'process_low'

    input:
    path error
    file x

    output:
    stdout

    script:
    wrongLog = file("${projectDir}/wrong.log")
    if (wrongLog.exists()){
        wrongLog.withReader{
            String line
            while (line = it.readLine()){
                exit 1, line
            }
        }
        
    }else{
        System.out.println("Check data DONE! The data is checked, and will be predicted!")
    }

    """
    if [ -e "$x" ]; then
        cat $error
        cat $x
    else
        echo "Check data DONE! The data is checked, and will be predicted!"
    fi
    """
}

process printResult{
    tag "printPred"

    label 'process_high'

    input:
    path result
    file x

    output:
    stdout

    """
    if [ -e "$x" ]; then
        cat $result
        cat $x
    else
        echo "Eroor! Errors in model predictions!"
    fi
    """
}

// predict
process model{
    tag "predict"

    label 'process_high'

    output:
    path 'result.txt', emit: result

        // --ann-file ${projectDir}/test.json\
        // --img-prefix ${projectDir}/test_data \
    """
    echo ${params.result} > result.txt
    python ${projectDir}/model/tools/test.py \
        --show-dir ${projectDir}/result \
        --output-eval ${projectDir}/result.txt \
        --config ${projectDir}/model/configs/ComparisonDetectorDataset/cas_rram_gram_multi.py \
        --checkpoint ${projectDir}/checkpoints/cas_rram_gram_multi_epoch_24.pth
    """
}


workflow{
    envCheck()
    printLog(envCheck.out.error, file("${projectDir}/wrong.log")).view{it.trim()}
    model()
}
// process isExit{
//     tag "isExit"

//     label 'process_low'

//     input:
//     path checkInfo

//     """
//     if [ `grep -c "$checkInfo" "Check data DONE! The data is checked, and will be predicted!"` -ne '0' ];then
//         exit 1
//     fi
//     """
// }

// Process data
// process processData{
//     tag "processData"

//     label 'process_mid'

//     """
//     python ${projectDir}/data_process.py \
//         --work_path ${projectDir}
//     """
// }

