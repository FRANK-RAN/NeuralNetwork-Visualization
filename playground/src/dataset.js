"use strict";
/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
Object.defineProperty(exports, "__esModule", { value: true });
exports.classifyXORData = exports.classifyCircleData = exports.classifySpiralData = exports.regressGaussian = exports.regressPlane = exports.classifyTwoGaussData = exports.shuffle = void 0;
var d3 = require("d3");
/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
function shuffle(array) {
    var counter = array.length;
    var temp = 0;
    var index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = Math.floor(Math.random() * counter);
        // Decrease counter by 1
        counter--;
        // And swap the last element with it
        temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }
}
exports.shuffle = shuffle;
function classifyTwoGaussData(numSamples, noise) {
    var points = [];
    var varianceScale = d3.scaleLinear().domain([0, .5]).range([0.5, 4]);
    var variance = varianceScale(noise);
    function genGauss(cx, cy, label) {
        for (var i = 0; i < numSamples / 2; i++) {
            var x = normalRandom(cx, variance);
            var y = normalRandom(cy, variance);
            points.push({ x: x, y: y, label: label });
        }
    }
    genGauss(2, 2, 1); // Gaussian with positive examples.
    genGauss(-2, -2, -1); // Gaussian with negative examples.
    return points;
}
exports.classifyTwoGaussData = classifyTwoGaussData;
function regressPlane(numSamples, noise) {
    var radius = 6;
    var labelScale = d3.scaleLinear()
        .domain([-10, 10])
        .range([-1, 1]);
    var getLabel = function (x, y) { return labelScale(x + y); };
    var points = [];
    for (var i = 0; i < numSamples; i++) {
        var x = randUniform(-radius, radius);
        var y = randUniform(-radius, radius);
        var noiseX = randUniform(-radius, radius) * noise;
        var noiseY = randUniform(-radius, radius) * noise;
        var label = getLabel(x + noiseX, y + noiseY);
        points.push({ x: x, y: y, label: label });
    }
    return points;
}
exports.regressPlane = regressPlane;
function regressGaussian(numSamples, noise) {
    var points = [];
    var labelScale = d3.scaleLinear()
        .domain([0, 2])
        .range([1, 0])
        .clamp(true);
    var gaussians = [
        [-4, 2.5, 1],
        [0, 2.5, -1],
        [4, 2.5, 1],
        [-4, -2.5, -1],
        [0, -2.5, 1],
        [4, -2.5, -1]
    ];
    function getLabel(x, y) {
        // Choose the one that is maximum in abs value.
        var label = 0;
        gaussians.forEach(function (_a) {
            var cx = _a[0], cy = _a[1], sign = _a[2];
            var newLabel = sign * labelScale(dist({ x: x, y: y }, { x: cx, y: cy }));
            if (Math.abs(newLabel) > Math.abs(label)) {
                label = newLabel;
            }
        });
        return label;
    }
    var radius = 6;
    for (var i = 0; i < numSamples; i++) {
        var x = randUniform(-radius, radius);
        var y = randUniform(-radius, radius);
        var noiseX = randUniform(-radius, radius) * noise;
        var noiseY = randUniform(-radius, radius) * noise;
        var label = getLabel(x + noiseX, y + noiseY);
        points.push({ x: x, y: y, label: label });
    }
    ;
    return points;
}
exports.regressGaussian = regressGaussian;
function classifySpiralData(numSamples, noise) {
    var points = [];
    var n = numSamples / 2;
    function genSpiral(deltaT, label) {
        for (var i = 0; i < n; i++) {
            var r = i / n * 5;
            var t = 1.75 * i / n * 2 * Math.PI + deltaT;
            var x = r * Math.sin(t) + randUniform(-1, 1) * noise;
            var y = r * Math.cos(t) + randUniform(-1, 1) * noise;
            points.push({ x: x, y: y, label: label });
        }
    }
    genSpiral(0, 1); // Positive examples.
    genSpiral(Math.PI, -1); // Negative examples.
    return points;
}
exports.classifySpiralData = classifySpiralData;
function classifyCircleData(numSamples, noise) {
    var points = [];
    var radius = 5;
    function getCircleLabel(p, center) {
        return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
    }
    // Generate positive points inside the circle.
    for (var i = 0; i < numSamples / 2; i++) {
        var r = randUniform(0, radius * 0.5);
        var angle = randUniform(0, 2 * Math.PI);
        var x = r * Math.sin(angle);
        var y = r * Math.cos(angle);
        var noiseX = randUniform(-radius, radius) * noise;
        var noiseY = randUniform(-radius, radius) * noise;
        var label = getCircleLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
        points.push({ x: x, y: y, label: label });
    }
    // Generate negative points outside the circle.
    for (var i = 0; i < numSamples / 2; i++) {
        var r = randUniform(radius * 0.7, radius);
        var angle = randUniform(0, 2 * Math.PI);
        var x = r * Math.sin(angle);
        var y = r * Math.cos(angle);
        var noiseX = randUniform(-radius, radius) * noise;
        var noiseY = randUniform(-radius, radius) * noise;
        var label = getCircleLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
        points.push({ x: x, y: y, label: label });
    }
    return points;
}
exports.classifyCircleData = classifyCircleData;
function classifyXORData(numSamples, noise) {
    function getXORLabel(p) { return p.x * p.y >= 0 ? 1 : -1; }
    var points = [];
    for (var i = 0; i < numSamples; i++) {
        var x = randUniform(-5, 5);
        var padding = 0.3;
        x += x > 0 ? padding : -padding; // Padding.
        var y = randUniform(-5, 5);
        y += y > 0 ? padding : -padding;
        var noiseX = randUniform(-5, 5) * noise;
        var noiseY = randUniform(-5, 5) * noise;
        var label = getXORLabel({ x: x + noiseX, y: y + noiseY });
        points.push({ x: x, y: y, label: label });
    }
    return points;
}
exports.classifyXORData = classifyXORData;
/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a, b) {
    return Math.random() * (b - a) + a;
}
/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean, variance) {
    if (mean === void 0) { mean = 0; }
    if (variance === void 0) { variance = 1; }
    var v1, v2, s;
    do {
        v1 = 2 * Math.random() - 1;
        v2 = 2 * Math.random() - 1;
        s = v1 * v1 + v2 * v2;
    } while (s > 1);
    var result = Math.sqrt(-2 * Math.log(s) / s) * v1;
    return mean + Math.sqrt(variance) * result;
}
/** Returns the eucledian distance between two points in space. */
function dist(a, b) {
    var dx = a.x - b.x;
    var dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}
