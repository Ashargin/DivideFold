вь
О▌
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceИ
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
П
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8с├
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
~
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
:*
dtype0
Ж
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/v/dense_3/kernel

)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes

:*
dtype0
Ж
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/m/dense_3/kernel

)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes

:*
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:*
dtype0
З
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╠*&
shared_nameAdam/v/dense_2/kernel
А
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes
:	╠*
dtype0
З
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╠*&
shared_nameAdam/m/dense_2/kernel
А
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes
:	╠*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╠*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	╠*
dtype0
Ц
serving_default_input_2Placeholder*5
_output_shapes#
!:                  ╠*
dtype0**
shape!:                  ╠
Б
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_55065570

NoOpNoOp
╦+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ж+
value№*B∙* BЄ*
╬
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
е
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_random_generator* 
ж
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
О
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
 
0
1
,2
-3*
 
0
1
,2
-3*
* 
░
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

9trace_0
:trace_1* 

;trace_0
<trace_1* 
* 
Б
=
_variables
>_iterations
?_learning_rate
@_index_dict
A
_momentums
B_velocities
C_update_step_xla*

Dserving_default* 
* 
* 
* 
С
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Jtrace_0
Ktrace_1* 

Ltrace_0
Mtrace_1* 
* 

0
1*

0
1*
* 
У
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

Ztrace_0
[trace_1* 

\trace_0
]trace_1* 
* 

,0
-1*

,0
-1*
* 
У
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

ctrace_0* 

dtrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

jtrace_0* 

ktrace_0* 
* 
.
0
1
2
3
4
5*

l0*
* 
* 
* 
* 
* 
* 
C
>0
m1
n2
o3
p4
q5
r6
s7
t8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
m0
o1
q2
s3*
 
n0
p1
r2
t3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
u	variables
v	keras_api
	wtotal
	xcount*
`Z
VARIABLE_VALUEAdam/m/dense_2/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_2/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_2/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_2/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_3/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_3/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_3/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_3/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*

w0
x1*

u	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┴
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	iterationlearning_rateAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_save_55065839
╝
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	iterationlearning_rateAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference__traced_restore_55065896═ь
Т
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065597

inputs

identity_1\
IdentityIdentityinputs*
T0*5
_output_shapes#
!:                  ╠i

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:                  ╠"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  ╠:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
и
¤
E__inference_dense_2_layer_call_and_return_conditional_losses_55065389

inputs4
!tensordot_readvariableop_resource:	╠-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	╠*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Г
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:                  ╠К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  ]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :                  n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :                  V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  ╠: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
ж

c
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065721

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:                  a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ў

f
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065659

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?q
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :                  Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Щ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :                  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=│
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    а
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*4
_output_shapes"
 :                  n
IdentityIdentitydropout/SelectV2:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
█
H
,__inference_dropout_2_layer_call_fn_55065580

inputs
identity└
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╠* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065465n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:                  ╠"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  ╠:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
ЖВ
║
!__inference__traced_save_55065839
file_prefix8
%read_disablecopyonread_dense_2_kernel:	╠3
%read_1_disablecopyonread_dense_2_bias:9
'read_2_disablecopyonread_dense_3_kernel:3
%read_3_disablecopyonread_dense_3_bias:,
"read_4_disablecopyonread_iteration:	 0
&read_5_disablecopyonread_learning_rate: A
.read_6_disablecopyonread_adam_m_dense_2_kernel:	╠A
.read_7_disablecopyonread_adam_v_dense_2_kernel:	╠:
,read_8_disablecopyonread_adam_m_dense_2_bias::
,read_9_disablecopyonread_adam_v_dense_2_bias:A
/read_10_disablecopyonread_adam_m_dense_3_kernel:A
/read_11_disablecopyonread_adam_v_dense_3_kernel:;
-read_12_disablecopyonread_adam_m_dense_3_bias:;
-read_13_disablecopyonread_adam_v_dense_3_bias:)
read_14_disablecopyonread_total: )
read_15_disablecopyonread_count: 
savev2_const
identity_33ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 в
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_2_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	╠*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	╠b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	╠y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 б
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_2_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 з
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 б
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Ъ
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_iteration^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ю
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_learning_rate^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: В
Read_6/DisableCopyOnReadDisableCopyOnRead.read_6_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 п
Read_6/ReadVariableOpReadVariableOp.read_6_disablecopyonread_adam_m_dense_2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	╠*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	╠f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	╠В
Read_7/DisableCopyOnReadDisableCopyOnRead.read_7_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 п
Read_7/ReadVariableOpReadVariableOp.read_7_disablecopyonread_adam_v_dense_2_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	╠*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	╠f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	╠А
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 и
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_adam_m_dense_2_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_9/DisableCopyOnReadDisableCopyOnRead,read_9_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 и
Read_9/ReadVariableOpReadVariableOp,read_9_disablecopyonread_adam_v_dense_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:Д
Read_10/DisableCopyOnReadDisableCopyOnRead/read_10_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_10/ReadVariableOpReadVariableOp/read_10_disablecopyonread_adam_m_dense_3_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_11/DisableCopyOnReadDisableCopyOnRead/read_11_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_11/ReadVariableOpReadVariableOp/read_11_disablecopyonread_adam_v_dense_3_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:В
Read_12/DisableCopyOnReadDisableCopyOnRead-read_12_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 л
Read_12/ReadVariableOpReadVariableOp-read_12_disablecopyonread_adam_m_dense_3_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 л
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_adam_v_dense_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_14/DisableCopyOnReadDisableCopyOnReadread_14_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_14/ReadVariableOpReadVariableOpread_14_disablecopyonread_total^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_15/DisableCopyOnReadDisableCopyOnReadread_15_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_15/ReadVariableOpReadVariableOpread_15_disablecopyonread_count^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: ╤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
valueЁBэB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHП
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B ├
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_32Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_33IdentityIdentity_32:output:0^NoOp*
T0*
_output_shapes
: я
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_33Identity_33:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:3/
-
_user_specified_nameAdam/v/dense_3/bias:3/
-
_user_specified_nameAdam/m/dense_3/bias:51
/
_user_specified_nameAdam/v/dense_3/kernel:51
/
_user_specified_nameAdam/m/dense_3/kernel:3
/
-
_user_specified_nameAdam/v/dense_2/bias:3	/
-
_user_specified_nameAdam/m/dense_2/bias:51
/
_user_specified_nameAdam/v/dense_2/kernel:51
/
_user_specified_nameAdam/m/dense_2/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
в
№
E__inference_dense_3_layer_call_and_return_conditional_losses_55065438

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  c
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :                  g
IdentityIdentitySigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :                  V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Т
▓
E__inference_model_1_layer_call_and_return_conditional_losses_55065485
input_2#
dense_2_55065467:	╠
dense_2_55065469:"
dense_3_55065478:
dense_3_55065480:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCall╦
dropout_2/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╠* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065465Ы
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_55065467dense_2_55065469*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_55065389ы
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065476Ы
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_55065478dense_3_55065480*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_55065438ч
flatten_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065455z
IdentityIdentity"flatten_1/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:                  f
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ╠: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:($
"
_user_specified_name
55065480:($
"
_user_specified_name
55065478:($
"
_user_specified_name
55065469:($
"
_user_specified_name
55065467:^ Z
5
_output_shapes#
!:                  ╠
!
_user_specified_name	input_2
в
№
E__inference_dense_3_layer_call_and_return_conditional_losses_55065704

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  c
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :                  g
IdentityIdentitySigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :                  V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
и	
╦
&__inference_signature_wrapper_55065570
input_2
unknown:	╠
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_55065343x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ╠: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
55065566:($
"
_user_specified_name
55065564:($
"
_user_specified_name
55065562:($
"
_user_specified_name
55065560:^ Z
5
_output_shapes#
!:                  ╠
!
_user_specified_name	input_2
л
Ш
*__inference_dense_2_layer_call_fn_55065606

inputs
unknown:	╠
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_55065389|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  ╠: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
55065602:($
"
_user_specified_name
55065600:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
щL
╥	
$__inference__traced_restore_55065896
file_prefix2
assignvariableop_dense_2_kernel:	╠-
assignvariableop_1_dense_2_bias:3
!assignvariableop_2_dense_3_kernel:-
assignvariableop_3_dense_3_bias:&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: ;
(assignvariableop_6_adam_m_dense_2_kernel:	╠;
(assignvariableop_7_adam_v_dense_2_kernel:	╠4
&assignvariableop_8_adam_m_dense_2_bias:4
&assignvariableop_9_adam_v_dense_2_bias:;
)assignvariableop_10_adam_m_dense_3_kernel:;
)assignvariableop_11_adam_v_dense_3_kernel:5
'assignvariableop_12_adam_m_dense_3_bias:5
'assignvariableop_13_adam_v_dense_3_bias:#
assignvariableop_14_total: #
assignvariableop_15_count: 
identity_17ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╘
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
valueЁBэB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHТ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B є
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_6AssignVariableOp(assignvariableop_6_adam_m_dense_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_7AssignVariableOp(assignvariableop_7_adam_v_dense_2_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_8AssignVariableOp&assignvariableop_8_adam_m_dense_2_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_9AssignVariableOp&assignvariableop_9_adam_v_dense_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_10AssignVariableOp)assignvariableop_10_adam_m_dense_3_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_v_dense_3_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_m_dense_3_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_v_dense_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 п
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: °
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_17Identity_17:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:3/
-
_user_specified_nameAdam/v/dense_3/bias:3/
-
_user_specified_nameAdam/m/dense_3/bias:51
/
_user_specified_nameAdam/v/dense_3/kernel:51
/
_user_specified_nameAdam/m/dense_3/kernel:3
/
-
_user_specified_nameAdam/v/dense_2/bias:3	/
-
_user_specified_nameAdam/m/dense_2/bias:51
/
_user_specified_nameAdam/v/dense_2/kernel:51
/
_user_specified_nameAdam/m/dense_2/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╧
H
,__inference_flatten_1_layer_call_fn_55065709

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065455i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Е
e
,__inference_dropout_3_layer_call_fn_55065642

inputs
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065406|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╬	
╧
*__inference_model_1_layer_call_fn_55065511
input_2
unknown:	╠
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_55065485x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ╠: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
55065507:($
"
_user_specified_name
55065505:($
"
_user_specified_name
55065503:($
"
_user_specified_name
55065501:^ Z
5
_output_shapes#
!:                  ╠
!
_user_specified_name	input_2
┴U
Ї
#__inference__wrapped_model_55065343
input_2D
1model_1_dense_2_tensordot_readvariableop_resource:	╠=
/model_1_dense_2_biasadd_readvariableop_resource:C
1model_1_dense_3_tensordot_readvariableop_resource:=
/model_1_dense_3_biasadd_readvariableop_resource:
identityИв&model_1/dense_2/BiasAdd/ReadVariableOpв(model_1/dense_2/Tensordot/ReadVariableOpв&model_1/dense_3/BiasAdd/ReadVariableOpв(model_1/dense_3/Tensordot/ReadVariableOpo
model_1/dropout_2/IdentityIdentityinput_2*
T0*5
_output_shapes#
!:                  ╠Ы
(model_1/dense_2/Tensordot/ReadVariableOpReadVariableOp1model_1_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	╠*
dtype0h
model_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       А
model_1/dense_2/Tensordot/ShapeShape#model_1/dropout_2/Identity:output:0*
T0*
_output_shapes
::э╧i
'model_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"model_1/dense_2/Tensordot/GatherV2GatherV2(model_1/dense_2/Tensordot/Shape:output:0'model_1/dense_2/Tensordot/free:output:00model_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$model_1/dense_2/Tensordot/GatherV2_1GatherV2(model_1/dense_2/Tensordot/Shape:output:0'model_1/dense_2/Tensordot/axes:output:02model_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
model_1/dense_2/Tensordot/ProdProd+model_1/dense_2/Tensordot/GatherV2:output:0(model_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 model_1/dense_2/Tensordot/Prod_1Prod-model_1/dense_2/Tensordot/GatherV2_1:output:0*model_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 model_1/dense_2/Tensordot/concatConcatV2'model_1/dense_2/Tensordot/free:output:0'model_1/dense_2/Tensordot/axes:output:0.model_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
model_1/dense_2/Tensordot/stackPack'model_1/dense_2/Tensordot/Prod:output:0)model_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:└
#model_1/dense_2/Tensordot/transpose	Transpose#model_1/dropout_2/Identity:output:0)model_1/dense_2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  ╠║
!model_1/dense_2/Tensordot/ReshapeReshape'model_1/dense_2/Tensordot/transpose:y:0(model_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 model_1/dense_2/Tensordot/MatMulMatMul*model_1/dense_2/Tensordot/Reshape:output:00model_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         k
!model_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:i
'model_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"model_1/dense_2/Tensordot/concat_1ConcatV2+model_1/dense_2/Tensordot/GatherV2:output:0*model_1/dense_2/Tensordot/Const_2:output:00model_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
model_1/dense_2/TensordotReshape*model_1/dense_2/Tensordot/MatMul:product:0+model_1/dense_2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  Т
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
model_1/dense_2/BiasAddBiasAdd"model_1/dense_2/Tensordot:output:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  }
model_1/dense_2/ReluRelu model_1/dense_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  Й
model_1/dropout_3/IdentityIdentity"model_1/dense_2/Relu:activations:0*
T0*4
_output_shapes"
 :                  Ъ
(model_1/dense_3/Tensordot/ReadVariableOpReadVariableOp1model_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
model_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       А
model_1/dense_3/Tensordot/ShapeShape#model_1/dropout_3/Identity:output:0*
T0*
_output_shapes
::э╧i
'model_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"model_1/dense_3/Tensordot/GatherV2GatherV2(model_1/dense_3/Tensordot/Shape:output:0'model_1/dense_3/Tensordot/free:output:00model_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$model_1/dense_3/Tensordot/GatherV2_1GatherV2(model_1/dense_3/Tensordot/Shape:output:0'model_1/dense_3/Tensordot/axes:output:02model_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
model_1/dense_3/Tensordot/ProdProd+model_1/dense_3/Tensordot/GatherV2:output:0(model_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 model_1/dense_3/Tensordot/Prod_1Prod-model_1/dense_3/Tensordot/GatherV2_1:output:0*model_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 model_1/dense_3/Tensordot/concatConcatV2'model_1/dense_3/Tensordot/free:output:0'model_1/dense_3/Tensordot/axes:output:0.model_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
model_1/dense_3/Tensordot/stackPack'model_1/dense_3/Tensordot/Prod:output:0)model_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┐
#model_1/dense_3/Tensordot/transpose	Transpose#model_1/dropout_3/Identity:output:0)model_1/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  ║
!model_1/dense_3/Tensordot/ReshapeReshape'model_1/dense_3/Tensordot/transpose:y:0(model_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 model_1/dense_3/Tensordot/MatMulMatMul*model_1/dense_3/Tensordot/Reshape:output:00model_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         k
!model_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:i
'model_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"model_1/dense_3/Tensordot/concat_1ConcatV2+model_1/dense_3/Tensordot/GatherV2:output:0*model_1/dense_3/Tensordot/Const_2:output:00model_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
model_1/dense_3/TensordotReshape*model_1/dense_3/Tensordot/MatMul:product:0+model_1/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  Т
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
model_1/dense_3/BiasAddBiasAdd"model_1/dense_3/Tensordot:output:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  Г
model_1/dense_3/SigmoidSigmoid model_1/dense_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  p
model_1/flatten_1/ShapeShapemodel_1/dense_3/Sigmoid:y:0*
T0*
_output_shapes
::э╧o
%model_1/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_1/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_1/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
model_1/flatten_1/strided_sliceStridedSlice model_1/flatten_1/Shape:output:0.model_1/flatten_1/strided_slice/stack:output:00model_1/flatten_1/strided_slice/stack_1:output:00model_1/flatten_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!model_1/flatten_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         л
model_1/flatten_1/Reshape/shapePack(model_1/flatten_1/strided_slice:output:0*model_1/flatten_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ж
model_1/flatten_1/ReshapeReshapemodel_1/dense_3/Sigmoid:y:0(model_1/flatten_1/Reshape/shape:output:0*
T0*0
_output_shapes
:                  z
IdentityIdentity"model_1/flatten_1/Reshape:output:0^NoOp*
T0*0
_output_shapes
:                  ╩
NoOpNoOp'^model_1/dense_2/BiasAdd/ReadVariableOp)^model_1/dense_2/Tensordot/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp)^model_1/dense_3/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ╠: : : : 2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2T
(model_1/dense_2/Tensordot/ReadVariableOp(model_1/dense_2/Tensordot/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2T
(model_1/dense_3/Tensordot/ReadVariableOp(model_1/dense_3/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
5
_output_shapes#
!:                  ╠
!
_user_specified_name	input_2
О
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065476

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :                  h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :                  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Й
e
,__inference_dropout_2_layer_call_fn_55065575

inputs
identityИвStatefulPartitionedCall╨
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╠* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065357}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ╠<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  ╠22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
╬	
╧
*__inference_model_1_layer_call_fn_55065498
input_2
unknown:	╠
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_55065458x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ╠: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
55065494:($
"
_user_specified_name
55065492:($
"
_user_specified_name
55065490:($
"
_user_specified_name
55065488:^ Z
5
_output_shapes#
!:                  ╠
!
_user_specified_name	input_2
и
¤
E__inference_dense_2_layer_call_and_return_conditional_losses_55065637

inputs4
!tensordot_readvariableop_resource:	╠-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	╠*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Г
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:                  ╠К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  ]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :                  n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :                  V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  ╠: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
ж

c
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065455

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:                  a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
О
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065664

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :                  h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :                  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ў

f
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065406

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?q
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :                  Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Щ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :                  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=│
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    а
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*4
_output_shapes"
 :                  n
IdentityIdentitydropout/SelectV2:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Т
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065465

inputs

identity_1\
IdentityIdentityinputs*
T0*5
_output_shapes#
!:                  ╠i

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:                  ╠"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  ╠:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
З
·
E__inference_model_1_layer_call_and_return_conditional_losses_55065458
input_2#
dense_2_55065390:	╠
dense_2_55065392:"
dense_3_55065439:
dense_3_55065441:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCall█
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╠* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065357г
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_55065390dense_2_55065392*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_55065389Я
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065406г
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_55065439dense_3_55065441*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_55065438ч
flatten_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065455z
IdentityIdentity"flatten_1/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:                  о
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  ╠: : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:($
"
_user_specified_name
55065441:($
"
_user_specified_name
55065439:($
"
_user_specified_name
55065392:($
"
_user_specified_name
55065390:^ Z
5
_output_shapes#
!:                  ╠
!
_user_specified_name	input_2
╫
H
,__inference_dropout_3_layer_call_fn_55065647

inputs
identity┐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065476m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
¤

f
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065357

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?r
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:                  ╠Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Ъ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:                  ╠*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=┤
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:                  ╠T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    б
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*5
_output_shapes#
!:                  ╠o
IdentityIdentitydropout/SelectV2:output:0*
T0*5
_output_shapes#
!:                  ╠"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  ╠:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
¤

f
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065592

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?r
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:                  ╠Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Ъ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:                  ╠*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=┤
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:                  ╠T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    б
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*5
_output_shapes#
!:                  ╠o
IdentityIdentitydropout/SelectV2:output:0*
T0*5
_output_shapes#
!:                  ╠"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  ╠:] Y
5
_output_shapes#
!:                  ╠
 
_user_specified_nameinputs
и
Ч
*__inference_dense_3_layer_call_fn_55065673

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_55065438|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
55065669:($
"
_user_specified_name
55065667:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs"зL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*├
serving_defaultп
I
input_2>
serving_default_input_2:0                  ╠F
	flatten_19
StatefulPartitionedCall:0                  tensorflow/serving/predict:│П
х
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
╝
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╝
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_random_generator"
_tf_keras_layer
╗
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
е
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
<
0
1
,2
-3"
trackable_list_wrapper
<
0
1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╟
9trace_0
:trace_12Р
*__inference_model_1_layer_call_fn_55065498
*__inference_model_1_layer_call_fn_55065511╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z9trace_0z:trace_1
¤
;trace_0
<trace_12╞
E__inference_model_1_layer_call_and_return_conditional_losses_55065458
E__inference_model_1_layer_call_and_return_conditional_losses_55065485╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z;trace_0z<trace_1
╬B╦
#__inference__wrapped_model_55065343input_2"Ш
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ь
=
_variables
>_iterations
?_learning_rate
@_index_dict
A
_momentums
B_velocities
C_update_step_xla"
experimentalOptimizer
,
Dserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
┐
Jtrace_0
Ktrace_12И
,__inference_dropout_2_layer_call_fn_55065575
,__inference_dropout_2_layer_call_fn_55065580й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zJtrace_0zKtrace_1
ї
Ltrace_0
Mtrace_12╛
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065592
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065597й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zLtrace_0zMtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ф
Strace_02╟
*__inference_dense_2_layer_call_fn_55065606Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zStrace_0
 
Ttrace_02т
E__inference_dense_2_layer_call_and_return_conditional_losses_55065637Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zTtrace_0
!:	╠2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
┐
Ztrace_0
[trace_12И
,__inference_dropout_3_layer_call_fn_55065642
,__inference_dropout_3_layer_call_fn_55065647й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zZtrace_0z[trace_1
ї
\trace_0
]trace_12╛
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065659
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065664й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z\trace_0z]trace_1
"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ф
ctrace_02╟
*__inference_dense_3_layer_call_fn_55065673Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zctrace_0
 
dtrace_02т
E__inference_dense_3_layer_call_and_return_conditional_losses_55065704Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zdtrace_0
 :2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ц
jtrace_02╔
,__inference_flatten_1_layer_call_fn_55065709Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zjtrace_0
Б
ktrace_02ф
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065721Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zktrace_0
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
l0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
*__inference_model_1_layer_call_fn_55065498input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щBц
*__inference_model_1_layer_call_fn_55065511input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
E__inference_model_1_layer_call_and_return_conditional_losses_55065458input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
E__inference_model_1_layer_call_and_return_conditional_losses_55065485input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
_
>0
m1
n2
o3
p4
q5
r6
s7
t8"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
m0
o1
q2
s3"
trackable_list_wrapper
<
n0
p1
r2
t3"
trackable_list_wrapper
╡2▓п
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
╥B╧
&__inference_signature_wrapper_55065570input_2"Щ
Т▓О
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ
	jinput_2
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тB▀
,__inference_dropout_2_layer_call_fn_55065575inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
тB▀
,__inference_dropout_2_layer_call_fn_55065580inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065592inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065597inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╘B╤
*__inference_dense_2_layer_call_fn_55065606inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_dense_2_layer_call_and_return_conditional_losses_55065637inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тB▀
,__inference_dropout_3_layer_call_fn_55065642inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
тB▀
,__inference_dropout_3_layer_call_fn_55065647inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065659inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065664inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╘B╤
*__inference_dense_3_layer_call_fn_55065673inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_dense_3_layer_call_and_return_conditional_losses_55065704inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
,__inference_flatten_1_layer_call_fn_55065709inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065721inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
N
u	variables
v	keras_api
	wtotal
	xcount"
_tf_keras_metric
&:$	╠2Adam/m/dense_2/kernel
&:$	╠2Adam/v/dense_2/kernel
:2Adam/m/dense_2/bias
:2Adam/v/dense_2/bias
%:#2Adam/m/dense_3/kernel
%:#2Adam/v/dense_3/kernel
:2Adam/m/dense_3/bias
:2Adam/v/dense_3/bias
.
w0
x1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
:  (2total
:  (2countо
#__inference__wrapped_model_55065343Ж,->в;
4в1
/К,
input_2                  ╠
к ">к;
9
	flatten_1,К)
	flatten_1                  ╟
E__inference_dense_2_layer_call_and_return_conditional_losses_55065637~=в:
3в0
.К+
inputs                  ╠
к "9в6
/К,
tensor_0                  
Ъ б
*__inference_dense_2_layer_call_fn_55065606s=в:
3в0
.К+
inputs                  ╠
к ".К+
unknown                  ╞
E__inference_dense_3_layer_call_and_return_conditional_losses_55065704},-<в9
2в/
-К*
inputs                  
к "9в6
/К,
tensor_0                  
Ъ а
*__inference_dense_3_layer_call_fn_55065673r,-<в9
2в/
-К*
inputs                  
к ".К+
unknown                  ╩
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065592Aв>
7в4
.К+
inputs                  ╠
p
к ":в7
0К-
tensor_0                  ╠
Ъ ╩
G__inference_dropout_2_layer_call_and_return_conditional_losses_55065597Aв>
7в4
.К+
inputs                  ╠
p 
к ":в7
0К-
tensor_0                  ╠
Ъ д
,__inference_dropout_2_layer_call_fn_55065575tAв>
7в4
.К+
inputs                  ╠
p
к "/К,
unknown                  ╠д
,__inference_dropout_2_layer_call_fn_55065580tAв>
7в4
.К+
inputs                  ╠
p 
к "/К,
unknown                  ╠╚
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065659}@в=
6в3
-К*
inputs                  
p
к "9в6
/К,
tensor_0                  
Ъ ╚
G__inference_dropout_3_layer_call_and_return_conditional_losses_55065664}@в=
6в3
-К*
inputs                  
p 
к "9в6
/К,
tensor_0                  
Ъ в
,__inference_dropout_3_layer_call_fn_55065642r@в=
6в3
-К*
inputs                  
p
к ".К+
unknown                  в
,__inference_dropout_3_layer_call_fn_55065647r@в=
6в3
-К*
inputs                  
p 
к ".К+
unknown                  └
G__inference_flatten_1_layer_call_and_return_conditional_losses_55065721u<в9
2в/
-К*
inputs                  
к "5в2
+К(
tensor_0                  
Ъ Ъ
,__inference_flatten_1_layer_call_fn_55065709j<в9
2в/
-К*
inputs                  
к "*К'
unknown                  ╧
E__inference_model_1_layer_call_and_return_conditional_losses_55065458Е,-FвC
<в9
/К,
input_2                  ╠
p

 
к "5в2
+К(
tensor_0                  
Ъ ╧
E__inference_model_1_layer_call_and_return_conditional_losses_55065485Е,-FвC
<в9
/К,
input_2                  ╠
p 

 
к "5в2
+К(
tensor_0                  
Ъ и
*__inference_model_1_layer_call_fn_55065498z,-FвC
<в9
/К,
input_2                  ╠
p

 
к "*К'
unknown                  и
*__inference_model_1_layer_call_fn_55065511z,-FвC
<в9
/К,
input_2                  ╠
p 

 
к "*К'
unknown                  ╝
&__inference_signature_wrapper_55065570С,-IвF
в 
?к<
:
input_2/К,
input_2                  ╠">к;
9
	flatten_1,К)
	flatten_1                  