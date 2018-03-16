rule evaluate:
    input:
        SAVED_MODEL,
        predictions = '{sample}_predictions'
    output:
        metrics = '{sample}_metrics'
    shell:
        """
        rm -rf {output.metrics}
        python {DO} evaluate {CONFIG_ARG} --predictions_path {input.predictions} --results_path {output.metrics}
        """