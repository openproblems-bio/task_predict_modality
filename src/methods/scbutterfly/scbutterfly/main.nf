workflow run_wf {
  take: input_ch
  main:
  output_ch = input_ch
    | scbutterfly_train.run(
      fromState: ["input_train_mod1", "input_train_mod2", "input_test_mod1"],
      toState: ["input_model": "output"]
    )
    | scbutterfly_predict.run(
      fromState: ["input_train_mod1", "input_train_mod2", "input_test_mod1", "input_model"],
      toState: ["output": "output"]
    )
    | map { tup ->
      [tup[0], [output: tup[1].output]]
    }

  emit: output_ch
}
