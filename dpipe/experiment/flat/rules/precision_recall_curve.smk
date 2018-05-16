rule precision_recall_curve:
    input:
        predictions = '{sample}_predictions'
    output:
        results_path = '{sample}_prc.csv'
    shell:
        """
        python {DO} prc {CONFIG_ARG} --predictions_path {input.predictions} --results_path {output.results_path}
        """
