�	Mۿ���"@Mۿ���"@!Mۿ���"@	.�hMD@.�hMD@!.�hMD@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Mۿ���"@+��$Ί�?A�wE��!@Y؝�<��?*	�v���d@2U
Iterator::Model::ParallelMapV2�:���?!VF�D!@@)�:���?1VF�D!@@:Preprocessing2F
Iterator::Model��Xl���?!9 ���NJ@)j��Gq�?1��'�C[4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat[\�3�?�?!���Z<2@)�@.q䁘?1�2{�S�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��C�.�?!鰶s�7@)衶� �?1��N��(,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-�cyW=�?!9f �"@)-�cyW=�?19f �"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorXo�
��z?!%4C�y@)Xo�
��z?1%4C�y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipx�W�L�?!��%(�G@)�G��v?1���
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�[>���?!Ge���9@)���T�n?1��t]4@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9.�hMD@I�����EX@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	+��$Ί�?+��$Ί�?!+��$Ί�?      ��!       "      ��!       *      ��!       2	�wE��!@�wE��!@!�wE��!@:      ��!       B      ��!       J	؝�<��?؝�<��?!؝�<��?R      ��!       Z	؝�<��?؝�<��?!؝�<��?b      ��!       JCPU_ONLYY.�hMD@b q�����EX@Y      Y@q6�?���?"�
device�Your program is NOT input-bound because only 2.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 