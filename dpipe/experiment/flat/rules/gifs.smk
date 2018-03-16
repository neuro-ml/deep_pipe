rule gifs:
    input:
        predictions = '{sample}_predictions'
    output:
        gifs_path = '{sample}_gifs'
    shell:
        """
        rm -rf {output.gifs_path}
        python {DO} generate_gifs {CONFIG_ARG} --predictions_path {input.predictions} --gifs_path {output.gifs_path}
        """
