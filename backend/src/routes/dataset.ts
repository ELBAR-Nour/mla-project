import express from 'express';
import { body, param, query, validationResult } from 'express-validator';
import { MLService } from '../services/ml-service';
import { WebSocketService } from '../services/websocket-service';

export const datasetRoutes = (mlService: MLService, wsService: WebSocketService) => {
  const router = express.Router();

  // Initialize dataset
  router.post('/initialize', [
    body('dataset_name').optional().isString(),
    body('split').optional().isString(),
    body('sample_size').optional().isInt({ min: 1 }),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const { dataset_name = 'PneumoniaMNIST', split = 'train', sample_size } = req.body;

      const result = await mlService.initializeDataset(dataset_name, split, sample_size);

      wsService.broadcast('dataset', {
        action: 'initialized',
        dataset: result,
      });

      res.json(result);
    } catch (error: any) {
      console.error('Dataset initialization error:', error);
      res.status(500).json({
        error: 'Failed to initialize dataset',
        details: error.message,
      });
    }
  });

  // Get dataset info
  router.get('/info', async (req: express.Request, res: express.Response) => {
    try {
      const info = await mlService.getDatasetInfo();
      res.json(info);
    } catch (error: any) {
      console.error('Dataset info error:', error);
      res.status(500).json({
        error: 'Failed to get dataset info',
        details: error.message,
      });
    }
  });

  // Get single sample
  router.get('/sample/:imageId', [
    param('imageId').isInt({ min: 0 }),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const imageId = parseInt(req.params.imageId);
      const sample = await mlService.getSample(imageId);
      res.json(sample);
    } catch (error: any) {
      console.error('Sample retrieval error:', error);
      res.status(500).json({
        error: 'Failed to get sample',
        details: error.message,
      });
    }
  });

  // Get batch of samples
  router.get('/batch', [
    query('count').optional().isInt({ min: 1, max: 200 }),
  ], async (req: express.Request, res: express.Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
      }

      const count = parseInt(req.query.count as string) || 10;
      const batch = await mlService.getBatch(count);
      res.json(batch);
    } catch (error: any) {
      console.error('Batch retrieval error:', error);
      res.status(500).json({
        error: 'Failed to get batch',
        details: error.message,
      });
    }
  });

  return router;
};
