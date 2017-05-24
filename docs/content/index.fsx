(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#I "../../bin"

(**
Brahma.FSharp.Examples
======================

Examples of Brahma.FSharp usage. 

Brahma.FSharp
-------------

Brahma.FSharp is aimed to creation tool for heterogeneous systems (multicore CPU + many GPGPUs) programming. 
F# quotation translator is used for integration with GPGPU, and wide range of F# primitives for parallel an 
asynchronous programming (MailboxProcessor, async, Array.Paralle, etc) simplifies utilization of heterogeneous 
systems.

<div class="row">
  <div class="span1"></div>
  <div class="span6">
    <div class="well well-small" id="nuget">
      The Brahma.FSharp library can be <a href="https://nuget.org/packages/Brahma.FSharp">installed from NuGet</a>:
      <pre>PM> Install-Package Brahma.FSharp</pre>
    </div>
  </div>
  <div class="span1"></div>
</div>

Features of Brahma.FSharp:

 * We are aimed to translate native F# code to OpenCL with minimization of different wrappers and custom types.
 * We use OpenCL for communication with GPU. So, you can work not only with NVIDIA hardware but with any device, 
which support OpenCL (e.g. with AMD devices).
 * We support tuples and structures.
 * We can use strongly typed kernels from OpenCL code in F#.

Example
-------

This example demonstrates using a function defined in this library.
Following function computes the result of the addition of matrix 'mat2' to the matrix 'mat1'

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
See more examples [here](tutorial.html)

Samples & documentation
-----------------------

 * [Tutorial](tutorial.html) contains a further explanation of this sample library.

 * [API Reference](reference/index.html) contains automatically generated documentation for all types, modules
   and functions in the library. This includes additional brief samples on using most of the
   functions.
 
Contributing and copyright
--------------------------

The project is hosted on [GitHub][gh] where you can [report issues][issues], fork 
the project and submit pull requests. If you're adding a new public API, please also 
consider adding [samples][content] that can be turned into a documentation. You might
also want to read the [library design notes][readme] to understand how it works.

The library is available under Public Domain license, which allows modification and 
redistribution for both commercial and non-commercial purposes. For more information see the 
[License file][license] in the GitHub repository. 

  [content]: https://github.com/YaccConstructor/Brahma.FSharp.Examples/tree/master/docs/content
  [gh]: https://github.com/YaccConstructor/Brahma.FSharp.Examples
  [issues]: https://github.com/YaccConstructor/Brahma.FSharp.Examples/issues
  [readme]: https://github.com/YaccConstructor/Brahma.FSharp.Examples/blob/master/README.md
  [license]: https://github.com/YaccConstructor/Brahma.FSharp.Examples/blob/master/LICENSE.txt
*)
