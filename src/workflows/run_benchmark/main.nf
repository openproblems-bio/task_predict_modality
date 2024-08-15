
// the list of methods
methods = [
  mean_per_gene,
  random_predict,
  zeros,
  solution,
  knnr_py,
  knnr_r,
  lm,
  lmds_irlba_rf,
  guanlab_dengkw_pm
]

// the list of metrics
metrics = [
  correlation,
  mse
]

workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}

workflow run_wf {
  take:
  input_ch

  main:

  /***************************
   * RUN METHODS AND METRICS *
   ***************************/
  score_ch = input_ch

    // extract the dataset metadata
    | extract_metadata.run(
      key: "metadata_mod1",
      fromState: [input: "input_train_mod1"],
      toState: { id, output, state ->
        state + [
          dataset_uns_mod1: readYaml(output.output).uns
        ]
      }
    )

    | extract_metadata.run(
      key: "metadata_mod2",
      fromState: [input: "input_test_mod2"],
      toState: { id, output, state ->
        state + [
          dataset_uns_mod2: readYaml(output.output).uns
        ]
      }
    )

    | map{ id, state ->
      def rna_norm = state.dataset_uns_mod1.modality == "GEX" ? state.dataset_uns_mod1.normalization_id : state.dataset_uns_mod2.normalization_id
      [id, state + [rna_norm: rna_norm]]
    }

    | run_benchmark_fun(
      methods: methods,
      metrics: metrics,
      methodFilter: {id, state, comp ->
        def norm = state.rna_norm
        def pref = comp.config.info.preferred_normalization
        def norm_check = (norm == "log_cp10k" && pref == "counts") || norm == pref
        def method_check = !state.method_ids || state.method_ids.contains(comp.config.name)
        method_check && norm_check
      },
      methodFromState: { id, state, comp ->
        def new_args = [
          input_train_mod1: state.input_train_mod1,
          input_train_mod2: state.input_train_mod2,
          input_test_mod1: state.input_test_mod1
        ]
        if (comp.config.info.type == "control_method") {
          new_args.input_test_mod2 = state.input_test_mod2
        }
        new_args
      },
      methodToState: { id, output, state, comp ->
        state + [
          method_id: comp.config.name,
          method_output: output.output
        ]
      },
      metricFromState: [
        input_test_mod2: "input_test_mod2", 
        input_prediction: "method_output"
      ],
      metricToState: { id, output, state, comp ->
        state + [
          metric_id: comp.config.name,
          metric_output: output.output
        ]
      }
    )

    | joinStates { ids, states ->
      def score_uns = states.collect{it.score_uns}
      def score_uns_yaml_blob = toYamlBlob(score_uns)
      def score_uns_file = tempFile("score_uns.yaml")
      score_uns_file.write(score_uns_yaml_blob)
      
      ["output", [scores: score_uns_file]]
    }


  /******************************
   * GENERATE OUTPUT YAML FILES *
   ******************************/
  // create dataset, method and metric metadata files
  metadata_ch = input_ch
    | create_metadata_files(
      datasetFromState: [input: "input_train_mod1"],
      methods: methods,
      metrics: metrics,
      meta: meta
    )

  // merge all of the output data 
  output_ch = score_ch
    | mix(metadata_ch)
    | joinStates{ ids, states ->
      def mergedStates = states.inject([:]) { acc, m -> acc + m }
      [ids[0], mergedStates]
    }

  emit:
  output_ch
}


/**
 * HELPER FUNCTIONS
 * 
 * - run_benchmark_fun: runs a benchmark with a list of methods and metrics
 * - create_metadata_files: creates metadata files for the benchmark
 */


def run_benchmark_fun(args) {
  // required args
  def methods_ = args.methods
  def metrics_ = args.metrics
  def methodFilter = args.methodFilter
  def methodFromState = args.methodFromState
  def methodToState = args.methodToState
  def metricFromState = args.metricFromState
  def metricToState = args.metricToState

  assert methods_, "methods must be defined"
  assert metrics_, "metrics must be defined"
  assert methodFromState, "methodFromState must be defined"
  assert methodToState, "methodToState must be defined"
  assert metricFromState, "metricFromState must be defined"
  assert metricToState, "metricToState must be defined"
  if (!methodFilter) {
    methodFilter = { id, state, comp ->
      !state.method_ids || state.method_ids.contains(comp.config.name)
    }
  }

  // optional args
  def keyPrefix = args.keyPrefix ?: ""
  def methodAuto = args.methodAuto ?: [:]
  def metricAuto = args.metricAuto ?: [:]

  // add the key prefix to the method and metric names
  if (keyPrefix && keyPrefix != "") {
    methods_ = methods.collect{ method ->
      method.run(key: keyPrefix + method.config.name)
    }
    metrics_ = metrics.collect{ metric ->
      metric.run(key: keyPrefix + metric.config.name)
    }
  }

  workflow bench {
    take: input_ch

    main:
    output_ch = input_ch
      // run all methods
      | runEach(
        components: methods_,
        filter: { id, state, comp ->
          !state.method_ids || state.method_ids.contains(comp.config.name)
        },
        id: { id, state, comp ->
          id + "." + comp.config.name
        },
        fromState: methodFromState,
        toState: methodToState,
        auto: methodAuto
      )

      // run all metrics
      | runEach(
        components: metrics_,
        filter: { id, state, comp ->
          !state.metric_ids || state.metric_ids.contains(comp.config.name)
        },
        id: { id, state, comp ->
          id + "." + comp.config.name
        },
        fromState: metricFromState,
        toState: metricToState,
        auto: metricAuto
      )

      // extract the scores
      | extract_metadata.run(
        key: "${keyPrefix}score_uns",
        fromState: [input: "metric_output"],
        toState: { id, output, state ->
          state + [
            score_uns: readYaml(output.output).uns
          ]
        }
      )

    emit: output_ch
  }
  return bench
}


def create_metadata_files(args) {
  // required args
  def meta_ = args.meta
  def methods_ = args.methods
  def metrics_ = args.metrics
  def datasetFromState = args.datasetFromState

  assert meta_, "meta must be defined"
  assert methods_, "methods must be defined"
  assert metrics_, "metrics must be defined"
  assert datasetFromState, "datasetFromState must be defined"

  workflow metadata {
    take: input_ch

    main:
    output_ch = input_ch

      | map{ id, state ->
        [id, state + ["_meta": [join_id: id]]]
      }

      | extract_metadata.run(
        key: "dataset_uns",
        fromState: args.datasetFromState,
        toState: { id, output, state ->
          state + [
            dataset_info: readYaml(output.output).uns
          ]
        }
      )
    
      | joinStates { ids, states ->
        assert states.size() > 0, "no states found"
        assert states[0]._meta, "no _meta found in state[0]"
        assert states.every{it.dataset_info}, "not all states have dataset_info"

        // combine the dataset info into one file
        def dataset_uns = states.collect{it.dataset_info}
        def dataset_uns_yaml_blob = toYamlBlob(dataset_uns)
        def dataset_uns_file = tempFile("dataset_uns.yaml")
        dataset_uns_file.write(dataset_uns_yaml_blob)

        // store the method configs in a file
        def method_configs = methods_.collect{it.config}
        def method_configs_yaml_blob = toYamlBlob(method_configs)
        def method_configs_file = tempFile("method_configs.yaml")
        method_configs_file.write(method_configs_yaml_blob)

        // store the metric configs in a file
        def metric_configs = metrics_.collect{it.config}
        def metric_configs_yaml_blob = toYamlBlob(metric_configs)
        def metric_configs_file = tempFile("metric_configs.yaml")
        metric_configs_file.write(metric_configs_yaml_blob)

        def task_info_file = meta_.resources_dir.resolve("task_info.yaml")

        def new_state = [
          dataset_uns: dataset_uns_file,
          method_configs: method_configs_file,
          metric_configs: metric_configs_file,
          task_info: task_info_file,
          _meta: states[0]._meta
        ]
        ["output", new_state]
      }
    emit: output_ch
  }
  return metadata
}
