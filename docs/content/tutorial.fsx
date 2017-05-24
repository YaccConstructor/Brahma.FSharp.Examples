(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#I "../../bin"

(**
Usage examples
========================




ArrayExamples
-------------

 * Following function generates an array filled with Fibonacci numbers
*)

let Fib length =
    let localWorkSize = 20
    let platformName = "NVIDIA*"
    let deviceType = DeviceType.Default
    
    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message
   
    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)    

    let command = 
        <@
            fun (rng:_1D) (a:array<_>) ->
                for i in 2 .. length - 1 do
                    a.[i] <- a.[i - 1] + a.[i - 2]
        @>

    let a = Array.init (length) (fun i -> int64(1))

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d =(new _1D(length, localWorkSize))
    kernelPrepare d a

    let _ = commandQueue.Add(kernelRun())
    let _ = commandQueue.Add(a.ToHost provider).Finish()
    commandQueue.Dispose()
    provider.Dispose()
    provider.CloseAllBuffers()

    a
(**
 * Following function makes elements of the array arr to equal 0 in range of [firstElement, lastElement]
*)

let Nulify (arr: array<_>) length firstElement lastElement =
    let localWorkSize = 20
    let platformName = "NVIDIA*"
    let deviceType = DeviceType.Default
    
    if firstElement >= lastElement
    then failwith "Invalid indexes"

    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message
   
    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)    

    let command = 
        <@
            fun (rng:_1D) (a:array<_>) ->
                let i = rng.GlobalID0
                if (i >= firstElement) && (i <= lastElement)
                then a.[i] <- 0
        @>

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d =(new _1D(length, localWorkSize))
    kernelPrepare d arr

    let _ = commandQueue.Add(kernelRun())
    let _ = commandQueue.Add(arr.ToHost provider).Finish()
    commandQueue.Dispose()
    provider.Dispose()
    provider.CloseAllBuffers()

    arr

(**
 * Following function reverts the array arr
*)

let Reflect (arr: array<_>) length =
    let localWorkSize = 20
    let platformName = "NVIDIA*"
    let deviceType = DeviceType.Default
    
    if length <= 1 
    then failwith "Invalid length"
    
    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message
   
    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)    

    let command = 
        <@
            fun (rng:_1D) (a:array<_>) ->
                let i = rng.GlobalID0
                let buff = a.[i] 
                a.[i] <- a.[length - i]
                a.[length - i] <- buff
        @>

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d =(new _1D(length, localWorkSize))
    kernelPrepare d arr

    let _ = commandQueue.Add(kernelRun())
    let _ = commandQueue.Add(arr.ToHost provider).Finish()
    commandQueue.Dispose()
    provider.Dispose()
    provider.CloseAllBuffers()

    arr
(**

MatrixExamples
--------------

 * Following function computes the result of the addition of matrix 'mat2' to the matrix 'mat1'
*)

let Addition (mat1: array<float>) (mat2: array<float>) rows columns =

    let localWorkSize0 = 2
    let localWorkSize1 = 2
    let platformName = "NVIDIA*"
    let deviceType = DeviceType.Default        
    
    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message
   
    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)    

    let command = 
        <@
            fun (rng:_2D) (a:array<_>) (b:array<_>) (c:array<_>) ->
                let x = rng.GlobalID1
                let y = rng.GlobalID0                
                c.[y * columns + x] <- a.[y * columns + x] + b.[y * columns + x] 
        @>

    let a = mat1
    let b = mat2
    let res = Array.zeroCreate(rows * columns)

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d =(new _2D(rows, columns, localWorkSize0, localWorkSize1))
    kernelPrepare d a b res

    let _ = commandQueue.Add(kernelRun())
    let _ = commandQueue.Add(res.ToHost provider).Finish()
    commandQueue.Dispose()
    provider.Dispose()
    provider.CloseAllBuffers()

    res
    
(**
 * Following function computes the result of multiplying matrix 'mat' by constant
*)
let ConstantAndMatrixMultiply (mat: array<int64>) constant rows columns =
    
    let localWorkSize0 = 2
    let localWorkSize1 = 2
    let platformName = "NVIDIA*"
    let deviceType = DeviceType.Default
    
    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message
   
    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)    

    let command = 
        <@
            fun (rng:_2D) (a:array<_>) c ->
                let x = rng.GlobalID1
                let y = rng.GlobalID0                
                a.[y * columns + x] <- a.[y * columns + x] * c
        @>

    let a = mat
    let c = constant

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d =(new _2D(rows, columns, localWorkSize0, localWorkSize1))
    kernelPrepare d a c

    let _ = commandQueue.Add(kernelRun())
    let _ = commandQueue.Add(a.ToHost provider).Finish()
    commandQueue.Dispose()
    provider.Dispose()
    provider.CloseAllBuffers()

    a
(**
 * Following function computes upper triangular matrix from matrix 'mat'
*)

let TriMat (mat: array<float>) rows columns =
    
    let localWorkSize0 = 2
    let localWorkSize1 = 2
    let mutable size = rows
    let platformName = "NVIDIA*"
    let deviceType = DeviceType.Default
    
    if rows <> columns
    then failwith "Can not compute triangle matrix"

    for i in 0 .. rows - 1 do
        if mat.[i * rows + i] = 0.0
        then failwith "Can not compute triangle matrix"
    
    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message
   
    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)    

    let command = 
        <@
            fun (rng:_2D) (a: array<_>) ->                
                for y in 0 .. size - 2 do
                    for x in y + 1 .. size - 1 do
                        let k = a.[x * size + y] / a.[y * size + y]
                        for i in 0 .. size - 1 do
                            a.[x * size + i] <- a.[x * size + i] - a.[y * size + i] * k
        @>

    let a = mat

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d =(new _2D(rows, columns, localWorkSize0, localWorkSize1))
    kernelPrepare d a

    let _ = commandQueue.Add(kernelRun())
    let _ = commandQueue.Add(a.ToHost provider).Finish()
    commandQueue.Dispose()
    provider.Dispose()
    provider.CloseAllBuffers()

    a

(**
Some more info
*)