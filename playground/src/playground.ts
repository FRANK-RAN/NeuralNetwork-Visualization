import * as nn from "./nn";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

document.addEventListener('DOMContentLoaded', function() {
  console.log("DOM fully loaded and parsed");
  console.log("Script loaded");
  
  let mainWidth;
  // Function to fetch data from the backend
  async function fetchData(url: string) {
  try {
      const response = await fetch(url);
      if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }
      console.log("Clean")
      return await response.json();
    } catch (error) {
      console.error("Error fetching data:", error);
  }
}

// Function to normalize the data
function normalizeData(data: number[][]): number[][] {
  const maxVal = Math.max(...data.flat());
  return data.map(row => row.map(value => value / maxVal));
}

// Function to transform data for heatmap visualization
function transformDataForHeatmap(data: number[][]): any[] {
  return data.flatMap((row, rowIndex) =>
      row.map((value, colIndex) => ({ row: rowIndex, col: colIndex, value }))
  );
}

// Function to create a heatmap visualization using D3.js
function createHeatmap(data: any[], elementId: string) {
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };
  const width = 300 - margin.left - margin.right;
  const height = 300 - margin.top - margin.bottom;
  
  // Remove any existing SVG to avoid overlaps
  d3.select(`#${elementId}`).select("svg").remove();
  
  // Create SVG container
  const svg = d3.select(`#${elementId}`)
  .append("svg")
  .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);
                
                // Create a color scale
                const colorScale = d3.scaleSequential(d3.interpolateInferno)
                .domain([0, 1]); // Assuming normalized data
                
                // Create and position heatmap cells
                svg.selectAll()
                .data(data)
     .enter()
     .append("rect")
     .attr("x", d => d.col * 20)
     .attr("y", d => d.row * 20)
     .attr("width", 20)
     .attr("height", 20)
     .style("fill", d => colorScale(d.value));
}

  // Function to load and display data
  async function loadAndDisplayData() {
    const activationsUrl = 'http://localhost:8000/activations/';
    const gradientsUrl = 'http://localhost:8000/gradients/';

    const activationsData = await fetchData(activationsUrl);
    const gradientsData = await fetchData(gradientsUrl);
    console.log("activationsData", activationsData)
    if (activationsData && gradientsData) {
      // Process and visualize activations
      const normalizedActivations = normalizeData(activationsData);
        const heatmapDataActivations = transformDataForHeatmap(normalizedActivations);
        createHeatmap(heatmapDataActivations, 'activationsContainer');
        
        // Process and visualize gradients
        const normalizedGradients = normalizeData(gradientsData);
        const heatmapDataGradients = transformDataForHeatmap(normalizedGradients);
        createHeatmap(heatmapDataGradients, 'gradientsContainer');
      }
    }

  // More scrolling
  d3.select(".more button").on("click", function() {
    let position = 800;
    d3.transition()
      .duration(1000)
      .tween("scroll", scrollTween(position));
    });

  function scrollTween(offset) {
    return function() {
      let i = d3.interpolateNumber(window.pageYOffset ||
          document.documentElement.scrollTop, offset);
      return function(t) { scrollTo(0, i(t)); };
    };
  }

  const RECT_SIZE = 30;
  const BIAS_SIZE = 5;
  const NUM_SAMPLES_CLASSIFY = 500;
  const NUM_SAMPLES_REGRESS = 1200;
  const DENSITY = 100;

  enum HoverType {
    BIAS, WEIGHT
  }

  interface InputFeature {
    f: (x: number, y: number) => number;
    label?: string;
  }

  let INPUTS: {[name: string]: InputFeature} = {
    "x": {f: (x, y) => x, label: "X_1"},
    "y": {f: (x, y) => y, label: "X_2"},
    "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
    "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
    "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
    "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
    "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
  };

  let HIDABLE_CONTROLS = [
    ["Show test data", "showTestData"],
    ["Discretize output", "discretize"],
    ["Play button", "playButton"],
    ["Step button", "stepButton"],
    ["Reset button", "resetButton"],
    ["Learning rate", "learningRate"],
    ["Activation", "activation"],
    ["Regularization", "regularization"],
    ["Regularization rate", "regularizationRate"],
    ["Problem type", "problem"],
    ["Which dataset", "dataset"],
    ["Ratio train data", "percTrainData"],
    ["Noise level", "noise"],
    ["Batch size", "batchSize"],
    ["# of hidden layers", "numHiddenLayers"],
  ];

  class Player {
    private timerIndex = 0;
    private isPlaying = false;
    private callback: (isPlaying: boolean) => void = null;

    /** Plays/pauses the player. */
    playOrPause() {
      if (this.isPlaying) {
        this.isPlaying = false;
        this.pause();
      } else {
        this.isPlaying = true;
        if (iter === 0) {
          simulationStarted();
        }
        this.play();
      }
    }
    
    onPlayPause(callback: (isPlaying: boolean) => void) {
      this.callback = callback;
    }

    play() {
      this.pause();
      this.isPlaying = true;
      if (this.callback) {
        this.callback(this.isPlaying);
      }
      this.start(this.timerIndex);
    }

    pause() {
      this.timerIndex++;
      this.isPlaying = false;
      if (this.callback) {
        this.callback(this.isPlaying);
      }
    }
    
    private start(localTimerIndex: number) {
      d3.timer(() => {
        if (localTimerIndex < this.timerIndex) {
          return true;  // Done.
        }
        oneStep();
        return false;  // Not done.
      }, 0);
    }
  }

  let state = State.deserializeState();

  // Filter out inputs that are hidden.
  state.getHiddenProps().forEach(prop => {
    if (prop in INPUTS) {
      delete INPUTS[prop];
    }
  });

  let boundary: {[id: string]: number[][]} = {};
  let selectedNodeId: string = null;
  // Plot the heatmap.
  let xDomain: [number, number] = [-6, 6];
  let heatMap = new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
          {showAxes: true});
  console.log("heatMap", heatMap)
  let linkWidthScale = d3.scaleLinear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);

  let colorScale = d3.scaleLinear<string, number>()
  .domain([-1, 0, 1])
  .range(["#f59322", "#e8eaeb", "#0877bd"])
  .clamp(true);

  let iter = 0;
  let trainData: Example2D[] = [];
  let testData: Example2D[] = [];
  let network: nn.Node[][] = null;
  let lossTrain = 0;
  let lossTest = 0;
  let player = new Player();
  let lineChart = new AppendingLineChart(d3.select("#linechart"),
      ["#777", "black"]);

      
      function makeGUI() {
    d3.select("#reset-button").on("click", () => {
      reset();
      userHasInteracted();
      d3.select("#play-pause-button");
    });

    d3.select("#play-pause-button").on("click", function () {
      // Change the button's content.
      userHasInteracted();
      player.playOrPause();
    });

    player.onPlayPause(isPlaying => {
      d3.select("#play-pause-button").classed("playing", isPlaying);
    });
    
    d3.select("#next-step-button").on("click", () => {
      player.pause();
      userHasInteracted();
      if (iter === 0) {
        simulationStarted();
      }
      oneStep();
    });
    
    d3.select("#data-regen-button").on("click", () => {
      generateData();
      parametersChanged = true;
    });

    let dataThumbnails = d3.selectAll("canvas[data-dataset]");
    dataThumbnails.on("click", function() {
      let element = this as HTMLCanvasElement;
      let newDataset = datasets[element.dataset.dataset];
      if (newDataset === state.dataset) {
        return; // No-op.
      }
      state.dataset = newDataset;
      dataThumbnails.classed("selected", false);
      d3.select(this).classed("selected", true);
      generateData();
      parametersChanged = true;
      reset();
    });
    
    
    let datasetKey = getKeyFromValue(datasets, state.dataset);
    // Select the dataset according to the current state.
    d3.select(`canvas[data-dataset=${datasetKey}]`)
      .classed("selected", true);
      
    let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
    regDataThumbnails.on("click", function() {
      let element = this as HTMLCanvasElement;
      let newDataset = regDatasets[element.dataset.regdataset];
      if (newDataset === state.regDataset) {
        return; // No-op.
      }
      state.regDataset =  newDataset;
      regDataThumbnails.classed("selected", false);
      d3.select(this).classed("selected", true);
      generateData();
      parametersChanged = true;
      reset();
    });
    
    let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
    // Select the dataset according to the current state.
    d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
      .classed("selected", true);
      
    d3.select("#add-layers").on("click", () => {
      if (state.numHiddenLayers >= 6) {
        return;
      }
      state.networkShape[state.numHiddenLayers] = 2;
      state.numHiddenLayers++;
      parametersChanged = true;
      reset();
    });

    d3.select("#remove-layers").on("click", () => {
      if (state.numHiddenLayers <= 0) {
        return;
      }
      state.numHiddenLayers--;
      state.networkShape.splice(state.numHiddenLayers);
      parametersChanged = true;
      reset();
    });
    
    let showTestData = d3.select("#show-test-data").on("change", function() {
      state.showTestData = (this as HTMLInputElement).checked;
      state.serialize();
      userHasInteracted();
      heatMap.updateTestPoints(state.showTestData ? testData : []);
    });
    // Check/uncheck the checkbox according to the current state.
    showTestData.property("checked", state.showTestData);
    
    let discretize = d3.select("#discretize").on("change", function() {
      state.discretize = (this as HTMLInputElement).checked;
      state.serialize();
      userHasInteracted();
      updateUI();
    });
    // Check/uncheck the checbox according to the current state.
    discretize.property("checked", state.discretize);
    
    let percTrain = d3.select("#percTrainData").on("input", function() {
      state.percTrainData = +(this as HTMLInputElement).value;
      d3.select("label[for='percTrainData'] .value").text((this as HTMLInputElement).value);
      generateData();
      parametersChanged = true;
      reset();
    });
    percTrain.property("value", state.percTrainData);
    d3.select("label[for='percTrainData'] .value").text(state.percTrainData);
    
    let noise = d3.select("#noise").on("input", function() {
      state.noise = +(this as HTMLInputElement).value;
      d3.select("label[for='noise'] .value").text((this as HTMLInputElement).value);
      generateData();
      parametersChanged = true;
      reset();
    });
    let currentMax = parseInt(noise.property("max"));
    if (state.noise > currentMax) {
      if (state.noise <= 80) {
        noise.property("max", state.noise);
      } else {
        state.noise = 50;
      }
    } else if (state.noise < 0) {
      state.noise = 0;
    }
    noise.property("value", state.noise);
    d3.select("label[for='noise'] .value").text(state.noise);
    
    let batchSize = d3.select("#batchSize").on("input", function() {
      state.batchSize = +(this as HTMLInputElement).value;
      d3.select("label[for='batchSize'] .value").text(+(this as HTMLInputElement).value);
      parametersChanged = true;
      reset();
    });
    batchSize.property("value", state.batchSize);
    d3.select("label[for='batchSize'] .value").text(state.batchSize);
    
    let activationDropdown = d3.select("#activations").on("change", function() {
      state.activation = activations[(this as HTMLInputElement).value];
      parametersChanged = true;
      reset();
    });
    activationDropdown.property("value",
        getKeyFromValue(activations, state.activation));

        let learningRate = d3.select("#learningRate").on("change", function() {
          state.learningRate = +(this as HTMLInputElement).value;
      state.serialize();
      userHasInteracted();
      parametersChanged = true;
    });
    learningRate.property("value", state.learningRate);
    
    let regularDropdown = d3.select("#regularizations").on("change",
        function() {
      state.regularization = regularizations[(this as HTMLInputElement).value];
      parametersChanged = true;
      reset();
    });
    regularDropdown.property("value",
        getKeyFromValue(regularizations, state.regularization));

        let regularRate = d3.select("#regularRate").on("change", function() {
      state.regularizationRate = +(this as HTMLInputElement).value;
      parametersChanged = true;
      reset();
    });
    regularRate.property("value", state.regularizationRate);
    
    let problem = d3.select("#problem").on("change", function() {
      state.problem = problems[(this as HTMLInputElement).value];
      generateData();
      drawDatasetThumbnails();
      parametersChanged = true;
      reset();
    });
    problem.property("value", getKeyFromValue(problems, state.problem));
    
    // Add scale to the gradient color map.
    let x = d3.scaleLinear().domain([-1, 1]).range([0, 144]);
    let xAxis = d3.axisBottom(this.xScale).tickValues([-1, 0, 1]).tickFormat(d3.format("d"));
    
    d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
      .attr("transform", "translate(0,10)")
      .call(xAxis);
      
      // Listen for css-responsive changes and redraw the svg network.

    window.addEventListener("resize", () => {
      let newWidth = document.querySelector("#main-part")
          .getBoundingClientRect().width;
      if (newWidth !== mainWidth) {
        mainWidth = newWidth;
        drawNetwork(network);
        updateUI(true);
      }
    });

    // Hide the text below the visualization depending on the URL.
    if (state.hideText) {
      d3.select("#article-text").style("display", "none");
      d3.select("div.more").style("display", "none");
      d3.select("header").style("display", "none");
    }
  }

  function updateBiasesUI(network: nn.Node[][]) {
    nn.forEachNode(network, true, node => {
      d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
    });
  }

  function updateWeightsUI(network: nn.Node[][], container) {
    console.log(network)
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      let currentLayer = network[layerIdx];
      // Update all the nodes in this layer.
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        for (let j = 0; j < node.inputLinks.length; j++) {
          let link = node.inputLinks[j];
          container.select(`#link${link.source.id}-${link.dest.id}`)
          .style({
                "stroke-dashoffset": -iter / 3,
                "stroke-width": linkWidthScale(Math.abs(link.weight)),
                "stroke": colorScale(link.weight)
              })
              .datum(link);
            }
          }
    }
  }

  function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
      container, node?: nn.Node) {
    let x = cx - RECT_SIZE / 2;
    let y = cy - RECT_SIZE / 2;
    
    let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
      });
      
      // Draw the main rectangle.
      nodeGroup.append("rect")
      .attr({
        x: 0,
        y: 0,
        width: RECT_SIZE,
        height: RECT_SIZE,
      });
      let activeOrNotClass = state[nodeId] ? "active" : "inactive";
      if (isInput) {
        let label = INPUTS[nodeId].label != null ?
          INPUTS[nodeId].label : nodeId;
      // Draw the input label.
      let text = nodeGroup.append("text").attr({
        class: "main-label",
        x: -10,
        y: RECT_SIZE / 2, "text-anchor": "end"
      });
      if (/[_^]/.test(label)) {
        let myRe = /(.*?)([_^])(.)/g;
        let myArray;
        let lastIndex;
        while ((myArray = myRe.exec(label)) != null) {
          lastIndex = myRe.lastIndex;
          let prefix = myArray[1];
          let sep = myArray[2];
          let suffix = myArray[3];
          if (prefix) {
            text.append("tspan").text(prefix);
          }
          text.append("tspan")
          .attr("baseline-shift", sep === "_" ? "sub" : "super")
          .style("font-size", "9px")
          .text(suffix);
        }
        if (label.substring(lastIndex)) {
          text.append("tspan").text(label.substring(lastIndex));
        }
      } else {
        text.append("tspan").text(label);
      }
      nodeGroup.classed(activeOrNotClass, true);
    }
    if (!isInput) {
      // Draw the node's bias.
      nodeGroup.append("rect")
      .attr({
          id: `bias-${nodeId}`,
          x: -BIAS_SIZE - 2,
          y: RECT_SIZE - BIAS_SIZE + 3,
          width: BIAS_SIZE,
          height: BIAS_SIZE,
        }).on("mouseenter", function() {
          updateHoverCard(HoverType.BIAS, node, d3.pointer(event, container.node()));
        }).on("mouseleave", function() {
          updateHoverCard(null);
        });
      }
      
      // Draw the node's canvas.
      let div = d3.select("#network").insert("div", ":first-child")
      .attr("id", `canvas-${nodeId}`)
      .attr("class", "canvas")
      .style("position", "absolute")
      .style("left", `${x + 3}px`)
      .style("top", `${y + 3}px`)
      .on("mouseenter", function() {
        selectedNodeId = nodeId;
        div.classed("hovered", true);
        nodeGroup.classed("hovered", true);
        updateDecisionBoundary(network, false);
        heatMap.updateBackground(boundary[nodeId], state.discretize);
      })
      .on("mouseleave", function() {
        selectedNodeId = null;
        div.classed("hovered", false);
        nodeGroup.classed("hovered", false);
        updateDecisionBoundary(network, false);
        heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
            state.discretize);
      });
      if (isInput) {
        div.on("click", function() {
          state[nodeId] = !state[nodeId];
          parametersChanged = true;
        reset();
      });
      div.style("cursor", "pointer");
    }
    if (isInput) {
      div.classed(activeOrNotClass, true);
    }
    let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
        xDomain, div, {noSvg: true});
    div.datum({heatmap: nodeHeatMap, id: nodeId});

  }

  // Draw network
  function drawNetwork(network: nn.Node[][]): void {
    let svg = d3.select("#svg");
    // Remove all svg elements.
    svg.select("g.core").remove();
    // Remove all div elements.
    d3.select("#network").selectAll("div.canvas").remove();
    d3.select("#network").selectAll("div.plus-minus-neurons").remove();
    
    // Get the width of the svg container.
    let padding = 3;
    let co = d3.select(".column.output").node() as HTMLDivElement;
    let cf = d3.select(".column.features").node() as HTMLDivElement;
    let width = co.offsetLeft - cf.offsetLeft;
    svg.attr("width", width);
    
    // Map of all node coordinates.
    let node2coord: {[id: string]: {cx: number, cy: number}} = {};
    let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
    // Draw the network layer by layer.
    let numLayers = network.length;
    let featureWidth = 118;
    let layerScale = d3.scaleOrdinal<number, number>()
        .domain(d3.range(1, numLayers - 1))
        // .rangePoints([featureWidth, width - RECT_SIZE], 0.7);      
    // let layerScale = d3.scale.ordinal<number, number>()
    //     .domain(d3.range(1, numLayers - 1))
    //     .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
    let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);
    
    
    let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
    let calloutWeights = d3.select(".callout.weights").style("display", "none");
    let idWithCallout = null;
    let targetIdWithCallout = null;
    
    // Draw the input layer separately.
    let cx = RECT_SIZE / 2 + 50;
    let nodeIds = Object.keys(INPUTS);
    let maxY = nodeIndexScale(nodeIds.length);
    nodeIds.forEach((nodeId, i) => {
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[nodeId] = {cx, cy};
      drawNode(cx, cy, nodeId, true, container);
    });

    // Draw the intermediate layers.
    for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
      let numNodes = network[layerIdx].length;
      let cx = layerScale(layerIdx) + RECT_SIZE / 2;
      maxY = Math.max(maxY, nodeIndexScale(numNodes));
      addPlusMinusControl(layerScale(layerIdx), layerIdx);
      for (let i = 0; i < numNodes; i++) {
        let node = network[layerIdx][i];
        let cy = nodeIndexScale(i) + RECT_SIZE / 2;
        node2coord[node.id] = {cx, cy};
        drawNode(cx, cy, node.id, false, container, node);
        
        // Show callout to thumbnails.
        let numNodes = network[layerIdx].length;
        let nextNumNodes = network[layerIdx + 1].length;
        if (idWithCallout == null &&
            i === numNodes - 1 &&
            nextNumNodes <= numNodes) {
          calloutThumb.style("display", null)
            .style("top", `${20 + 3 + cy}px`)
            .style("left", `${cx}px`);
            idWithCallout = node.id;
        }

        // Draw links.
        for (let j = 0; j < node.inputLinks.length; j++) {
          let link = node.inputLinks[j];
          let path: SVGPathElement = drawLink(link, node2coord, network,
              container, j === 0, j, node.inputLinks.length).node() as any;
          // Show callout to weights.
          let prevLayer = network[layerIdx - 1];
          let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
          if (targetIdWithCallout == null &&
              i === numNodes - 1 &&
              link.source.id === lastNodePrevLayer.id &&
              (link.source.id !== idWithCallout || numLayers <= 5) &&
              link.dest.id !== idWithCallout &&
              prevLayer.length >= numNodes) {
            let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
            calloutWeights.style("display", null)
              .style("top", `${midPoint.y + 5}px`)
              .style("left", `${midPoint.x + 3}px`);
              targetIdWithCallout = link.dest.id;
            }
        }
      }
    }
    
    // Draw the output node separately.
    cx = width + RECT_SIZE / 2;
    let node = network[numLayers - 1][0];
    let cy = nodeIndexScale(0) + RECT_SIZE / 2;
    node2coord[node.id] = {cx, cy};
    // Draw links.
    for (let i = 0; i < node.inputLinks.length; i++) {
      let link = node.inputLinks[i];
      drawLink(link, node2coord, network, container, i === 0, i,
          node.inputLinks.length);
    }
    // Adjust the height of the svg.
    svg.attr("height", maxY);
    
    // Adjust the height of the features column.
    let height = Math.max(
      getRelativeHeight(calloutThumb),
      getRelativeHeight(calloutWeights),
      getRelativeHeight(d3.select("#network"))
      );
    d3.select(".column.features").style("height", height + "px");
  }

  function getRelativeHeight(selection) {
    let node = selection.node() as HTMLAnchorElement;
    return node.offsetHeight + node.offsetTop;
  }

  function addPlusMinusControl(x: number, layerIdx: number) {
    let div = d3.select("#network").append("div")
      .classed("plus-minus-neurons", true)
      .style("left", `${x - 10}px`);
      
      let i = layerIdx - 1;
      let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
      firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
          let numNeurons = state.networkShape[i];
          if (numNeurons >= 8) {
            return;
          }
          state.networkShape[i]++;
          parametersChanged = true;
          reset();
        })
        .append("i")
        .attr("class", "material-icons")
        .text("add");

        firstRow.append("button")
        .attr("class", "mdl-button mdl-js-button mdl-button--icon")
        .on("click", () => {
          let numNeurons = state.networkShape[i];
          if (numNeurons <= 1) {
            return;
          }
          state.networkShape[i]--;
          parametersChanged = true;
          reset();
        })
        .append("i")
        .attr("class", "material-icons")
        .text("remove");
        
    let suffix = state.networkShape[i] > 1 ? "s" : "";
    div.append("div").text(
      state.networkShape[i] + " neuron" + suffix
    );
  }

  function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
      coordinates?: [number, number]) {
    let hovercard = d3.select("#hovercard");
    if (type == null) {
      hovercard.style("display", "none");
      d3.select("#svg").on("click", null);
      return;
    }
    d3.select("#svg").on("click", () => {
      hovercard.select(".value").style("display", "none");
      let input = hovercard.select("input");
      input.style("display", null);
      input.on("input", function() {
        if ((this as HTMLInputElement).value != null && (this as HTMLInputElement).value !== "") {
          if (type === HoverType.WEIGHT) {
            (nodeOrLink as nn.Link).weight = +(this as HTMLInputElement).value;
          } else {
            (nodeOrLink as nn.Node).bias = +(this as HTMLInputElement).value;
          }
          updateUI();
        }
      });
      input.on("keypress", () => {
        if ((event as KeyboardEvent).key === "Enter") {
          updateHoverCard(type, nodeOrLink, coordinates);
        }
      });
      (input.node() as HTMLInputElement).focus();
    });
    let value = (type === HoverType.WEIGHT) ?
      (nodeOrLink as nn.Link).weight :
      (nodeOrLink as nn.Node).bias;
      let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
      hovercard.style("display", "block")
      .style("top", `${coordinates[1]}px`)
      .style("left", `${coordinates[0] + 20}px`);
      hovercard.select(".type").text(name);
      hovercard.select(".value")
      .style("display", null)
      .text(value.toPrecision(2));
    hovercard.select("input")
      .property("value", value.toPrecision(2))
      .style("display", "none");
    }

  function drawLink(
      input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
      network: nn.Node[][], container,
      isFirst: boolean, index: number, length: number) {
    let line = container.insert("path", ":first-child");
    let source = node2coord[input.source.id];
    let dest = node2coord[input.dest.id];
    let datum = {
      source: {
        y: source.cx + RECT_SIZE / 2 + 2,
        x: source.cy
      },
      target: {
        y: dest.cx - RECT_SIZE / 2,
        x: dest.cy + ((index - (length - 1) / 2) / length) * 12
      }
    };
    // Define the diagonal line generator
    let diagonal = d3.line<[number, number]>()  // Specify the type for the line generator
    .x(d => d[0])
      .y(d => d[1])
      .curve(d3.curveLinear);

      // Create the data array for the line generator
    let lineData: [number, number][] = [  // Explicitly declare as an array of tuples
      [datum.source.y, datum.source.x], // Start point
      [datum.target.y, datum.target.x]  // End point
    ];

    // Use the line generator to create the 'd' attribute
    line.attr("d", diagonal(lineData));
    
    // Add an invisible thick link that will be used for
    // showing the weight value on hover.
    container.append("path")
      .attr("d", diagonal(lineData))
      .attr("class", "link-hover")
      .on("mouseenter", function() {
        updateHoverCard(HoverType.WEIGHT, input, d3.pointer(event, this));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
      return line;
    }

  /**
   * Given a neural network, it asks the network for the output (prediction)
   * of every node in the network using inputs sampled on a square grid.
   * It returns a map where each key is the node ID and the value is a square
   * matrix of the outputs of the network for each input in the grid respectively.
  */
  function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
    if (firstTime) {
      boundary = {};
      nn.forEachNode(network, true, node => {
        boundary[node.id] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId] = new Array(DENSITY);
      }
    }
    let xScale = d3.scaleLinear().domain([0, DENSITY - 1]).range(xDomain);
    let yScale = d3.scaleLinear().domain([DENSITY - 1, 0]).range(xDomain);  
    // let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
    // let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

    let i = 0, j = 0;
    for (i = 0; i < DENSITY; i++) {
      if (firstTime) {
        nn.forEachNode(network, true, node => {
          boundary[node.id][i] = new Array(DENSITY);
        });
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i] = new Array(DENSITY);
        }
      }
      for (j = 0; j < DENSITY; j++) {
        // 1 for points inside the circle, and 0 for points outside the circle.
        let x = xScale(i);
        let y = yScale(j);
        let input = constructInput(x, y);
        nn.forwardProp(network, input);
        nn.forEachNode(network, true, node => {
          boundary[node.id][i][j] = node.output;
        });
        if (firstTime) {
          // Go through all predefined inputs.
          for (let nodeId in INPUTS) {
            boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
          }
        }
      }
    }
  }

  function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
    let loss = 0;
    for (let i = 0; i < dataPoints.length; i++) {
      let dataPoint = dataPoints[i];
      let input = constructInput(dataPoint.x, dataPoint.y);
      let output = nn.forwardProp(network, input);
      loss += nn.Errors.SQUARE.error(output, dataPoint.label);
    }
    return loss / dataPoints.length;
  }

  function updateUI(firstStep = false) {
    // Update the links visually.
    updateWeightsUI(network, d3.select("g.core"));
    // Update the bias values visually.
    updateBiasesUI(network);
    // Get the decision boundary of the network.
    updateDecisionBoundary(network, firstStep);
    let selectedId = selectedNodeId != null ?
        selectedNodeId : nn.getOutputNode(network).id;
    heatMap.updateBackground(boundary[selectedId], state.discretize);
    
    // Update all decision boundaries.
    d3.select("#network").selectAll("div.canvas")
    .each(function(data: {heatmap: HeatMap, id: string}) {
      data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
          state.discretize);
    });
    
    function zeroPad(n: number): string {
      let pad = "000000";
      return (pad + n).slice(-pad.length);
    }
    
    function addCommas(s: string): string {
      return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
    
    function humanReadable(n: number): string {
      return n.toFixed(3);
    }
    
    // Update loss and iteration number.
    d3.select("#loss-train").text(humanReadable(lossTrain));
    d3.select("#loss-test").text(humanReadable(lossTest));
    d3.select("#iter-number").text(addCommas(zeroPad(iter)));
    lineChart.addDataPoint([lossTrain, lossTest]);
  }

  function constructInputIds(): string[] {
    let result: string[] = [];
    for (let inputName in INPUTS) {
      if (state[inputName]) {
        result.push(inputName);
      }
    }
    return result;
  }

  function constructInput(x: number, y: number): number[] {
    let input: number[] = [];
    for (let inputName in INPUTS) {
      if (state[inputName]) {
        input.push(INPUTS[inputName].f(x, y));
      }
    }
    return input;
  }

  function oneStep(): void {
    iter++;
    trainData.forEach((point, i) => {
      let input = constructInput(point.x, point.y);
      nn.forwardProp(network, input);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
      if ((i + 1) % state.batchSize === 0) {
        nn.updateWeights(network, state.learningRate, state.regularizationRate);
      }
    });
    // Compute the loss.
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
    updateUI();
  }

  // export function getOutputWeights(network: nn.Node[][]): number[] {
  //   let weights: number[] = [];
  //   for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
  //     let currentLayer = network[layerIdx];
  //     for (let i = 0; i < currentLayer.length; i++) {
  //       let node = currentLayer[i];
  //       for (let j = 0; j < node.outputs.length; j++) {
  //         let output = node.outputs[j];
  //         weights.push(output.weight);
  //       }
  //     }
  //   }
  //   return weights;
  // }

  function reset(onStartup=false) {
    lineChart.reset();
    state.serialize();
    if (!onStartup) {
      userHasInteracted();
    }
    player.pause();

    let suffix = state.numHiddenLayers !== 1 ? "s" : "";
    d3.select("#layers-label").text("Hidden layer" + suffix);
    d3.select("#num-layers").text(state.numHiddenLayers);
    
    // Make a simple network.
    iter = 0;
    let numInputs = constructInput(0 , 0).length;
    let shape = [numInputs].concat(state.networkShape).concat([1]);
    let outputActivation = (state.problem === Problem.REGRESSION) ?
        nn.Activations.LINEAR : nn.Activations.TANH;
    network = nn.buildNetwork(shape, state.activation, outputActivation,
        state.regularization, constructInputIds(), state.initZero);
        lossTrain = getLoss(network, trainData);
        lossTest = getLoss(network, testData);
        drawNetwork(network);
        updateUI(true);
      };

  function drawDatasetThumbnails() {
    function renderThumbnail(canvas, dataGenerator) {
      let w = 100;
      let h = 100;
      canvas.setAttribute("width", w);
      canvas.setAttribute("height", h);
      let context = canvas.getContext("2d");
      let data = dataGenerator(200, 0);
      data.forEach(function(d) {
        context.fillStyle = colorScale(d.label);
        context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
      });
      d3.select(canvas.parentNode).style("display", null);
    }
    d3.selectAll(".dataset").style("display", "none");
    
    if (state.problem === Problem.CLASSIFICATION) {
      for (let dataset in datasets) {
        let canvas: any =
            document.querySelector(`canvas[data-dataset=${dataset}]`);
            let dataGenerator = datasets[dataset];
        renderThumbnail(canvas, dataGenerator);
      }
    }
    if (state.problem === Problem.REGRESSION) {
      for (let regDataset in regDatasets) {
        let canvas: any =
        document.querySelector(`canvas[data-regDataset=${regDataset}]`);
        let dataGenerator = regDatasets[regDataset];
        renderThumbnail(canvas, dataGenerator);
      }
    }
  }

  function hideControls() {
    // Set display:none to all the UI elements that are hidden.
    let hiddenProps = state.getHiddenProps();
    hiddenProps.forEach(prop => {
      let controls = d3.selectAll(`.ui-${prop}`);
      if (controls.size() === 0) {
        console.warn(`0 html elements found with class .ui-${prop}`);
      }
      controls.style("display", "none");
    });
    
    // Also add checkbox for each hidable control in the "use it in classrom"
    // section.
    let hideControls = d3.select(".hide-controls");
    HIDABLE_CONTROLS.forEach(([text, id]) => {
      let label = hideControls.append("label")
        .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
      let input = label.append("input")
        .attr("type", "checkbox")
        .attr("class", "mdl-checkbox__input");
      if (hiddenProps.indexOf(id) === -1) {
        input.attr("checked", "true");
      }
      input.on("change", function() {
        state.setHideProperty(id, !this.checked);
        state.serialize();
        userHasInteracted();
        d3.select(".hide-controls-link")
        .attr("href", window.location.href);
      });
      label.append("span")
        .attr("class", "mdl-checkbox__label label")
        .text(text);
    });
    d3.select(".hide-controls-link")
      .attr("href", window.location.href);
  }

  function generateData(firstTime = false) {
    if (!firstTime) {
      // Change the seed.
      state.seed = Math.random().toFixed(5);
      state.serialize();
      userHasInteracted();
    }
    Math.seedrandom(state.seed);
    let numSamples = (state.problem === Problem.REGRESSION) ?
        NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
    let generator = state.problem === Problem.CLASSIFICATION ?
        state.dataset : state.regDataset;
    let data = generator(numSamples, state.noise / 100);
    // Shuffle the data in-place.
    shuffle(data);
    // Split into train and test data.
    let splitIndex = Math.floor(data.length * state.percTrainData / 100);
    trainData = data.slice(0, splitIndex);
    testData = data.slice(splitIndex);
    heatMap.updatePoints(trainData);
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  }

  let firstInteraction = true;
  let parametersChanged = false;

  function userHasInteracted() {
    if (!firstInteraction) {
      return;
    }
    firstInteraction = false;
    let page = 'index';
    if (state.tutorial != null && state.tutorial !== '') {
      page = `/v/tutorials/${state.tutorial}`;
    }
    ga('set', 'page', page);
    ga('send', 'pageview', {'sessionControl': 'start'});
  }

  function simulationStarted() {
    ga('send', {
      hitType: 'event',
      eventCategory: 'Starting Simulation',
      eventAction: parametersChanged ? 'changed' : 'unchanged',
      eventLabel: state.tutorial == null ? '' : state.tutorial
    });
    parametersChanged = false;
  }

  drawDatasetThumbnails();
  makeGUI();
  generateData(true);
  reset(true);
  hideControls();
  // Call this function at the appropriate place in your script
  loadAndDisplayData();
});