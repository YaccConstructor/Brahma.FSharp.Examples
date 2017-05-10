// Copyright (c) 2017 Kirill Smirenko <k.smirenko@gmail.com>
// All rights reserved.
// 
// The contents of this file are made available under the terms of the
// Eclipse Public License v1.0 (the "License") which accompanies this
// distribution, and is available at the following URL:
// http://www.opensource.org/licenses/eclipse-1.0.php
// 
// Software distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTY OF ANY KIND, either expressed or implied. See the License for
// the specific language governing rights and limitations under the License.
// 
// By using this software in any fashion, you are agreeing to be bound by the
// terms of the License.

module Brahma.FSharp.MatrixMultiplyTP

open OpenCL.Net
open Brahma.FSharp.OpenCL.Core
open Brahma.FSharp.OpenCL.Extensions
open Brahma.FSharp.OpenCL.TypeProvider.Provided
open Brahma.Helpers
open Brahma.OpenCL

// TP configuration
let constantsPath = __SOURCE_DIRECTORY__ + "/MyGEMM/constants.h"
let [<Literal>] clSourcePath = __SOURCE_DIRECTORY__ + "/MyGEMM/mygemm.cl"
type ProvidedType = KernelProvider<clSourcePath, TreatPointersAsArrays=true>

// utils
let random = new System.Random()
let makeFloat32Matrix rows cols =
    Array.init (rows * cols) (fun i -> random.NextDouble() |> float32)

// configuration
let matrixSizes = [128; 256; 512; 1024; 2048]
let iterations = 20
let deviceType = DeviceType.Default

let Run platformName =
    // load OpenCL C sources and headers
    let constants = System.IO.File.ReadAllText(constantsPath)
    let clSource = System.IO.File.ReadAllText(clSourcePath)

    // init compute resources
    let computeProvider =
        try ComputeProvider.Create(platformName, deviceType)
        with
        | ex -> failwith ex.Message

    let device = computeProvider.Devices |> Seq.head
    let lws, ex = OpenCL.Net.Cl.GetDeviceInfo(device, OpenCL.Net.DeviceInfo.MaxWorkGroupSize)
    let maxLocalWorkSize = int <| lws.CastTo<uint64>()

    let mutable commandQueue = new CommandQueue(computeProvider, device)

    // main loop - launching & time calculating
    printfn "Device: %A" computeProvider
    printfn "Will do %A iterations for all matrices." iterations
    for size in matrixSizes do
        printfn "Size %A:" size
        let localWorkSize = 8 //min maxLocalWorkSize size

        let aValues = makeFloat32Matrix size size
        let bValues = makeFloat32Matrix size size
        let cParallel = Array.zeroCreate(size * size)

        let commandFs =
            <@
                fun (r:_2D) (a:array<_>) (b:array<_>) (c:array<_>) -> 
                    let tx = r.GlobalID0
                    let ty = r.GlobalID1
                    let mutable buf = c.[ty * size + tx]
                    for k in 0 .. size - 1 do
                        buf <- buf + (a.[ty * size + k] * b.[k * size + tx])
                    c.[ty * size + tx] <- buf
            @>
        let command1 =
            <@
                fun (r:_2D) (a:array<_>) (b:array<_>) (c:array<_>) ->
                    ProvidedType.myGEMM1(size, size, size, a, b, c)
            @>
        let command2 =
            <@
                fun (r:_2D) (a:array<_>) (b:array<_>) (c:array<_>) ->
                    ProvidedType.myGEMM2(size, size, size, a, b, c)
            @>

        let configs =
            [
                (commandFs, [], "brahmaKernel");
                (command1, [constants; clSource], "myGEMM1");
                (command2, [constants; clSource], "myGEMM2")
            ]

        for command, addsrc, desc in configs do
            let outCode = ref ""
            let _, kernelPrepare, kernelRun = computeProvider.Compile(command, _outCode = outCode, _additionalSources = addsrc)
            let d =(new _2D(size, size, localWorkSize, localWorkSize))
            //let d =(new _2D(size, size))
            kernelPrepare d aValues bValues cParallel

            //printf "%12s:\t" desc
            try
                for i in 0 .. iterations - 1 do
                    Timer<string>.Global.Start()
                    let _ = commandQueue.Add(kernelRun()).Finish()
                    Timer<string>.Global.Lap("OpenCL")

                let _ = commandQueue.Add(cParallel.ToHost computeProvider).Finish()

                let avgTime = Timer<string>.Global.Average("OpenCL")
                Timer<string>.Global.Reset()
                printfn "%.8f" avgTime
            with
            | ex -> printfn "FAIL: %A" ex.Message

    printfn "Done."

    commandQueue.Dispose()
    computeProvider.Dispose()
    computeProvider.CloseAllBuffers()

Run "NVIDIA*" |> ignore
