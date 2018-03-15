rule train_model:
    input:
        TRAIN_IDS, VAL_IDS
    output:
        SAVED_MODEL, TRAINING_LOG
    shell:
        """
        rm -rf {SAVED_MODEL} {TRAINING_LOG}
        python {DO} train_model {CONFIG_ARG} --train_ids_path {TRAIN_IDS} --val_ids_path {VAL_IDS} --log_path {TRAINING_LOG} --save_model_path {SAVED_MODEL}
        """

rule predict:
    input:
        SAVED_MODEL,
        ids = '{sample}_ids.json'
    output:
        predictions = '{sample}_predictions'
    shell:
        """
        rm -rf {output.predictions}
        python {DO} predict {CONFIG_ARG} --ids_path {input.ids} --output_path {output.predictions} --restore_model_path {SAVED_MODEL}
        """