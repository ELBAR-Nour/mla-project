import express from 'express';
import { body, validationResult } from 'express-validator';
import { MLService } from '../services/ml-service';
import { WebSocketService } from '../services/websocket-service';

export const modelRoutes = (mlService: MLService, wsService: WebSocketService) => {
  const router = express.Router();

  // List available .pt artifacts
  router.get('/artifacts', async (req: express.Request, res: express.Response) => {
    try {
      const artifacts = await mlService.getModelArtifacts();
      res.json(artifacts);
    } catch (error: any) {
      console.error('Model artifact listing error:', error);
      const response = mlService.formatError(error, 'Failed to list model artifacts');
      res.status(response.status).json(response.body);
    }
  });

  // Load a classifier artifact
  router.post('/load', [
    body('model_name').isString().notEmpty(),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const result = await mlService.loadModel(req.body.model_name);

      wsService.broadcast('model', {
        action: 'loaded',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('Model loading error:', error);
      const response = mlService.formatError(error, 'Failed to load model');
      res.status(response.status).json(response.body);
    }
  });

  // Run classifier inference
  router.post('/inference', [
    body('model_name').optional().isString(),
    body('image_id').optional().isInt({ min: 0 }),
    body('image_base64').optional().isString(),
    body('dataset_name').optional().isString(),
    body('split').optional().isString(),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const result = await mlService.runInference(req.body);

      wsService.broadcast('model', {
        action: 'inference_completed',
        result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('Model inference error:', error);
      const response = mlService.formatError(error, 'Failed to run model inference');
      res.status(response.status).json(response.body);
    }
  });

  // Get model metrics
  router.get('/metrics', async (req: express.Request, res: express.Response) => {
    try {
      const metrics = await mlService.getModelMetrics();
      res.json(metrics);
    } catch (error: any) {
      console.error('Model metrics error:', error);
      const response = mlService.formatError(error, 'Failed to get model metrics');
      res.status(response.status).json(response.body);
    }
  });

  // Get confusion matrix
  router.get('/confusion-matrix', async (req: express.Request, res: express.Response) => {
    try {
      const cm = await mlService.getConfusionMatrix({
        model_name: typeof req.query.model_name === 'string' ? req.query.model_name : undefined,
        dataset_name: typeof req.query.dataset_name === 'string' ? req.query.dataset_name : undefined,
      });
      res.json(cm);
    } catch (error: any) {
      console.error('Confusion matrix error:', error);
      const response = mlService.formatError(error, 'Failed to get confusion matrix');
      res.status(response.status).json(response.body);
    }
  });

  // Get ROC curve
  router.get('/roc', async (req: express.Request, res: express.Response) => {
    try {
      const roc = await mlService.getROCCurve({
        model_name: typeof req.query.model_name === 'string' ? req.query.model_name : undefined,
        dataset_name: typeof req.query.dataset_name === 'string' ? req.query.dataset_name : undefined,
      });
      res.json(roc);
    } catch (error: any) {
      console.error('ROC curve error:', error);
      const response = mlService.formatError(error, 'Failed to get ROC curve');
      res.status(response.status).json(response.body);
    }
  });

  return router;
};
