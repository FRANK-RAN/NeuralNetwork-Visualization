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
exports.reduceMatrix = exports.HeatMap = void 0;
var d3 = require("d3");
/** Number of different shades (colors) when drawing a gradient heatmap */
var NUM_SHADES = 30;
/**
 * Draws a heatmap using canvas. Used for showing the learned decision
 * boundary of the classification algorithm. Can also draw data points
 * using an svg overlayed on top of the canvas heatmap.
 */
var HeatMap = /** @class */ (function () {
    function HeatMap(width, numSamples, xDomain, yDomain, container, userSettings) {
        this.settings = {
            showAxes: false,
            noSvg: false
        };
        this.numSamples = numSamples;
        var height = width;
        var padding = userSettings.showAxes ? 20 : 0;
        if (userSettings != null) {
            // overwrite the defaults with the user-specified settings.
            for (var prop in userSettings) {
                this.settings[prop] = userSettings[prop];
            }
        }
        this.xScale = d3.scaleLinear()
            .domain(xDomain)
            .range([0, width - 2 * padding]);
        this.yScale = d3.scaleLinear()
            .domain(yDomain)
            .range([height - 2 * padding, 0]);
        // Get a range of colors.
        var tmpScale = d3.scaleLinear()
            .domain([0, .5, 1])
            .range(["#f59322", "#e8eaeb", "#0877bd"])
            .clamp(true);
        // Due to numerical error, we need to specify
        // d3.range(0, end + small_epsilon, step)
        // in order to guarantee that we will have end/step entries with
        // the last element being equal to end.
        var colors = d3.range(0, 1 + 1E-9, 1 / NUM_SHADES).map(function (a) {
            return tmpScale(a);
        });
        this.color = d3.scaleQuantize()
            .domain([-1, 1])
            .range(colors);
        container = container.append("div")
            .style({
            width: "".concat(width, "px"),
            height: "".concat(height, "px"),
            position: "relative",
            top: "-".concat(padding, "px"),
            left: "-".concat(padding, "px")
        });
        this.canvas = container.append("canvas")
            .attr("width", numSamples)
            .attr("height", numSamples)
            .style("width", (width - 2 * padding) + "px")
            .style("height", (height - 2 * padding) + "px")
            .style("position", "absolute")
            .style("top", "".concat(padding, "px"))
            .style("left", "".concat(padding, "px"));
        if (!this.settings.noSvg) {
            this.svg = container.append("svg").attr({
                "width": width,
                "height": height
            }).style({
                // Overlay the svg on top of the canvas.
                "position": "absolute",
                "left": "0",
                "top": "0"
            }).append("g")
                .attr("transform", "translate(".concat(padding, ",").concat(padding, ")"));
            this.svg.append("g").attr("class", "train");
            this.svg.append("g").attr("class", "test");
        }
        if (this.settings.showAxes) {
            var xAxis = d3.axisBottom(this.xScale);
            var yAxis = d3.axisRight(this.yScale);
            this.svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0,".concat(height - 2 * padding, ")"))
                .call(xAxis);
            this.svg.append("g")
                .attr("class", "y axis")
                .attr("transform", "translate(" + (width - 2 * padding) + ",0)")
                .call(yAxis);
        }
    }
    HeatMap.prototype.updateTestPoints = function (points) {
        if (this.settings.noSvg) {
            throw Error("Can't add points since noSvg=true");
        }
        this.updateCircles(this.svg.select("g.test"), points);
    };
    HeatMap.prototype.updatePoints = function (points) {
        if (this.settings.noSvg) {
            throw Error("Can't add points since noSvg=true");
        }
        this.updateCircles(this.svg.select("g.train"), points);
    };
    HeatMap.prototype.updateBackground = function (data, discretize) {
        var dx = data[0].length;
        var dy = data.length;
        if (dx !== this.numSamples || dy !== this.numSamples) {
            throw new Error("The provided data matrix must be of size " +
                "numSamples X numSamples");
        }
        // Compute the pixel colors; scaled by CSS.
        var context = this.canvas.node().getContext("2d");
        var image = context.createImageData(dx, dy);
        for (var y = 0, p = -1; y < dy; ++y) {
            for (var x = 0; x < dx; ++x) {
                var value = data[x][y];
                if (discretize) {
                    value = (value >= 0 ? 1 : -1);
                }
                var c = d3.rgb(this.color(value));
                image.data[++p] = c.r;
                image.data[++p] = c.g;
                image.data[++p] = c.b;
                image.data[++p] = 160;
            }
        }
        context.putImageData(image, 0, 0);
    };
    HeatMap.prototype.updateCircles = function (container, points) {
        var _this = this;
        // Keep only points that are inside the bounds.
        var xDomain = this.xScale.domain();
        var yDomain = this.yScale.domain();
        points = points.filter(function (p) {
            return p.x >= xDomain[0] && p.x <= xDomain[1]
                && p.y >= yDomain[0] && p.y <= yDomain[1];
        });
        // Attach data to initially empty selection.
        var selection = container.selectAll("circle").data(points);
        // Insert elements to match length of points array.
        selection.enter().append("circle").attr("r", 3);
        // Update points to be in the correct position.
        selection
            .attr({
            cx: function (d) { return _this.xScale(d.x); },
            cy: function (d) { return _this.yScale(d.y); },
        })
            .style("fill", function (d) { return _this.color(d.label); });
        // Remove points if the length has gone down.
        selection.exit().remove();
    };
    return HeatMap;
}()); // Close class HeatMap.
exports.HeatMap = HeatMap;
function reduceMatrix(matrix, factor) {
    if (matrix.length !== matrix[0].length) {
        throw new Error("The provided matrix must be a square matrix");
    }
    if (matrix.length % factor !== 0) {
        throw new Error("The width/height of the matrix must be divisible by " +
            "the reduction factor");
    }
    var result = new Array(matrix.length / factor);
    for (var i = 0; i < matrix.length; i += factor) {
        result[i / factor] = new Array(matrix.length / factor);
        for (var j = 0; j < matrix.length; j += factor) {
            var avg = 0;
            // Sum all the values in the neighborhood.
            for (var k = 0; k < factor; k++) {
                for (var l = 0; l < factor; l++) {
                    avg += matrix[i + k][j + l];
                }
            }
            avg /= (factor * factor);
            result[i / factor][j / factor] = avg;
        }
    }
    return result;
}
exports.reduceMatrix = reduceMatrix;
