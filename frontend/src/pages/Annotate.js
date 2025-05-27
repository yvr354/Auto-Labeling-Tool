import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Button, 
  Typography,
  Row,
  Col,
  Select,
  Space,
  Divider,
  Upload,
  message,
  List,
  Tag,
  Input,
  Modal,
  Progress,
  Tooltip,
  Statistic
} from 'antd';
import {
  EditOutlined,
  RobotOutlined,
  SaveOutlined,
  StepForwardOutlined,
  StepBackwardOutlined,
  UndoOutlined,
  UploadOutlined,
  PlusOutlined,
  DeleteOutlined,
  DownloadOutlined,
  BulbOutlined,
  ThunderboltOutlined,
  EyeOutlined
} from '@ant-design/icons';
import SmartAnnotationInterface from '../components/SmartAnnotationInterface';

const { Title, Paragraph } = Typography;
const { Option } = Select;
const { Dragger } = Upload;

const Annotate = () => {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [currentImage, setCurrentImage] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [images, setImages] = useState([]);
  const [annotations, setAnnotations] = useState([]);
  const [classes, setClasses] = useState([
    { name: 'Person', color: '#ff4d4f' },
    { name: 'Car', color: '#52c41a' },
    { name: 'Bicycle', color: '#1890ff' },
    { name: 'Dog', color: '#fa8c16' },
    { name: 'Cat', color: '#722ed1' }
  ]);
  const [selectedClass, setSelectedClass] = useState(0);
  const [isAutoLabeling, setIsAutoLabeling] = useState(false);
  const [showClassModal, setShowClassModal] = useState(false);
  const [newClassName, setNewClassName] = useState('');
  const [annotationStats, setAnnotationStats] = useState({
    total: 0,
    annotated: 0,
    pending: 0
  });

  // Load sample images for demo
  useEffect(() => {
    // In a real app, this would load from the selected dataset
    const sampleImages = [
      {
        id: 1,
        name: 'sample1.jpg',
        url: 'https://via.placeholder.com/800x600/4CAF50/FFFFFF?text=Sample+Image+1',
        annotations: []
      },
      {
        id: 2,
        name: 'sample2.jpg', 
        url: 'https://via.placeholder.com/800x600/2196F3/FFFFFF?text=Sample+Image+2',
        annotations: []
      },
      {
        id: 3,
        name: 'sample3.jpg',
        url: 'https://via.placeholder.com/800x600/FF9800/FFFFFF?text=Sample+Image+3', 
        annotations: []
      }
    ];
    
    setImages(sampleImages);
    if (sampleImages.length > 0) {
      setCurrentImage(sampleImages[0]);
      setAnnotations(sampleImages[0].annotations);
    }
  }, []);

  // Update stats when annotations change
  useEffect(() => {
    const annotated = images.filter(img => img.annotations && img.annotations.length > 0).length;
    setAnnotationStats({
      total: images.length,
      annotated: annotated,
      pending: images.length - annotated
    });
  }, [images, annotations]);

  // Handle image navigation
  const navigateImage = (direction) => {
    const newIndex = direction === 'next' 
      ? Math.min(currentImageIndex + 1, images.length - 1)
      : Math.max(currentImageIndex - 1, 0);
    
    if (newIndex !== currentImageIndex) {
      // Save current annotations
      saveCurrentAnnotations();
      
      // Load new image
      setCurrentImageIndex(newIndex);
      setCurrentImage(images[newIndex]);
      setAnnotations(images[newIndex].annotations || []);
    }
  };

  // Save annotations for current image
  const saveCurrentAnnotations = () => {
    if (currentImage) {
      const updatedImages = images.map(img => 
        img.id === currentImage.id 
          ? { ...img, annotations: annotations }
          : img
      );
      setImages(updatedImages);
      message.success('Annotations saved!');
    }
  };

  // Handle annotations change
  const handleAnnotationsChange = (newAnnotations) => {
    setAnnotations(newAnnotations);
  };

  // Auto-label current image
  const handleAutoLabel = async () => {
    if (!selectedModel || !currentImage) {
      message.warning('Please select a model and image');
      return;
    }

    setIsAutoLabeling(true);
    try {
      // Simulate API call to auto-labeling service
      message.loading('Running AI auto-labeling...', 0);
      
      // Mock auto-labeling results
      setTimeout(() => {
        const mockAnnotations = [
          {
            type: 'bbox',
            bbox: { x: 100, y: 100, width: 200, height: 150 },
            classIndex: 0,
            confidence: 0.95,
            source: 'ai'
          },
          {
            type: 'polygon',
            points: [
              { x: 400, y: 200 },
              { x: 500, y: 180 },
              { x: 520, y: 280 },
              { x: 450, y: 300 },
              { x: 380, y: 250 }
            ],
            classIndex: 1,
            confidence: 0.87,
            source: 'ai'
          }
        ];
        
        setAnnotations([...annotations, ...mockAnnotations]);
        message.destroy();
        message.success(`Added ${mockAnnotations.length} AI annotations!`);
        setIsAutoLabeling(false);
      }, 2000);
      
    } catch (error) {
      message.destroy();
      message.error('Auto-labeling failed');
      setIsAutoLabeling(false);
    }
  };

  // Add new class
  const handleAddClass = () => {
    if (newClassName.trim()) {
      const colors = ['#ff4d4f', '#52c41a', '#1890ff', '#fa8c16', '#722ed1', '#eb2f96', '#13c2c2'];
      const newClass = {
        name: newClassName.trim(),
        color: colors[classes.length % colors.length]
      };
      setClasses([...classes, newClass]);
      setNewClassName('');
      setShowClassModal(false);
      message.success('Class added successfully!');
    }
  };

  // Delete class
  const handleDeleteClass = (index) => {
    const newClasses = classes.filter((_, i) => i !== index);
    setClasses(newClasses);
    if (selectedClass >= newClasses.length) {
      setSelectedClass(newClasses.length - 1);
    }
    message.success('Class deleted');
  };

  // Export annotations
  const handleExportAnnotations = () => {
    const exportData = {
      images: images.map(img => ({
        name: img.name,
        annotations: img.annotations
      })),
      classes: classes
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'annotations.json';
    a.click();
    URL.revokeObjectURL(url);
    message.success('Annotations exported!');
  };

  // Upload handler
  const uploadProps = {
    name: 'file',
    multiple: true,
    accept: 'image/*',
    beforeUpload: (file) => {
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        message.error('You can only upload image files!');
        return false;
      }
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      const newImage = {
        id: Date.now() + Math.random(),
        name: file.name,
        url: url,
        annotations: []
      };
      
      setImages(prev => [...prev, newImage]);
      message.success(`${file.name} uploaded successfully`);
      return false; // Prevent auto upload
    }
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '16px 24px', borderBottom: '1px solid #f0f0f0' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              🎯 Smart Annotation Tool
            </Title>
            <Paragraph style={{ margin: 0, color: '#666' }}>
              AI-powered annotation with manual refinement capabilities
            </Paragraph>
          </Col>
          <Col>
            <Space>
              <Button icon={<UploadOutlined />} onClick={() => document.querySelector('.ant-upload input').click()}>
                Upload Images
              </Button>
              <Button icon={<DownloadOutlined />} onClick={handleExportAnnotations}>
                Export
              </Button>
              <Button 
                type="primary" 
                icon={<ThunderboltOutlined />}
                onClick={handleAutoLabel}
                loading={isAutoLabeling}
                disabled={!currentImage || !selectedModel}
              >
                AI Auto-Label
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      <Row style={{ flex: 1, height: 0 }}>
        {/* Main Canvas Area */}
        <Col xs={24} lg={18} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: 16, flex: 1, display: 'flex', flexDirection: 'column' }}>
            {!currentImage ? (
              <Card style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Dragger {...uploadProps} style={{ padding: 40 }}>
                  <p className="ant-upload-drag-icon">
                    <EditOutlined style={{ fontSize: 64, color: '#1890ff' }} />
                  </p>
                  <p className="ant-upload-text">Click or drag images to start annotating</p>
                  <p className="ant-upload-hint">
                    Support for single or bulk image upload. JPG, PNG, GIF formats supported.
                  </p>
                </Dragger>
              </Card>
            ) : (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Image Navigation */}
                <Card size="small" style={{ marginBottom: 8 }}>
                  <Row justify="space-between" align="middle">
                    <Col>
                      <Space>
                        <span>{currentImage.name}</span>
                        <Tag color="blue">{currentImageIndex + 1} / {images.length}</Tag>
                        {annotations.length > 0 && (
                          <Tag color="green">{annotations.length} annotations</Tag>
                        )}
                      </Space>
                    </Col>
                    <Col>
                      <Space>
                        <Button 
                          icon={<StepBackwardOutlined />}
                          onClick={() => navigateImage('prev')}
                          disabled={currentImageIndex === 0}
                        />
                        <Button 
                          icon={<StepForwardOutlined />}
                          onClick={() => navigateImage('next')}
                          disabled={currentImageIndex === images.length - 1}
                        />
                        <Button 
                          type="primary"
                          icon={<SaveOutlined />}
                          onClick={saveCurrentAnnotations}
                        >
                          Save
                        </Button>
                      </Space>
                    </Col>
                  </Row>
                </Card>

                {/* Smart Annotation Interface */}
                <div style={{ flex: 1 }}>
                  <SmartAnnotationInterface
                    currentImage={currentImage}
                    annotations={annotations}
                    onAnnotationsChange={handleAnnotationsChange}
                    classes={classes}
                    selectedClass={selectedClass}
                    onClassChange={setSelectedClass}
                    onSave={saveCurrentAnnotations}
                    onAutoLabel={handleAutoLabel}
                  />
                </div>
              </div>
            )}
          </div>
        </Col>

        {/* Right Sidebar */}
        <Col xs={24} lg={6} style={{ height: '100%', borderLeft: '1px solid #f0f0f0' }}>
          <div style={{ padding: 16, height: '100%', overflow: 'auto' }}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              
              {/* Progress Stats */}
              <Card title="📊 Progress" size="small">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic 
                      title="Total" 
                      value={annotationStats.total}
                      valueStyle={{ color: '#1890ff' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic 
                      title="Done" 
                      value={annotationStats.annotated}
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic 
                      title="Pending" 
                      value={annotationStats.pending}
                      valueStyle={{ color: '#fa8c16' }}
                    />
                  </Col>
                </Row>
                <Progress 
                  percent={annotationStats.total > 0 ? Math.round((annotationStats.annotated / annotationStats.total) * 100) : 0}
                  style={{ marginTop: 16 }}
                />
              </Card>

              {/* Model Selection */}
              <Card title="🤖 AI Models" size="small">
                <Select
                  placeholder="Select AI model"
                  style={{ width: '100%', marginBottom: 12 }}
                  value={selectedModel}
                  onChange={setSelectedModel}
                >
                  <Option value="yolo11n">YOLO11 Nano (Fast)</Option>
                  <Option value="yolo11s">YOLO11 Small</Option>
                  <Option value="yolo11m">YOLO11 Medium</Option>
                  <Option value="yolo11l">YOLO11 Large (Accurate)</Option>
                  <Option value="sam">SAM (Segmentation)</Option>
                </Select>
                
                <Button 
                  type="primary" 
                  icon={<RobotOutlined />}
                  block
                  onClick={handleAutoLabel}
                  loading={isAutoLabeling}
                  disabled={!selectedModel || !currentImage}
                >
                  Run AI Auto-Label
                </Button>
              </Card>

              {/* Classes Management */}
              <Card 
                title="🏷️ Classes" 
                size="small"
                extra={
                  <Button 
                    type="text" 
                    icon={<PlusOutlined />}
                    onClick={() => setShowClassModal(true)}
                  />
                }
              >
                <List
                  size="small"
                  dataSource={classes}
                  renderItem={(cls, index) => (
                    <List.Item
                      style={{ 
                        padding: '8px 0',
                        backgroundColor: selectedClass === index ? '#f6ffed' : 'transparent',
                        borderRadius: 4,
                        cursor: 'pointer'
                      }}
                      onClick={() => setSelectedClass(index)}
                      actions={[
                        <Button 
                          type="text" 
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteClass(index);
                          }}
                        />
                      ]}
                    >
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <div 
                          style={{
                            width: 12,
                            height: 12,
                            backgroundColor: cls.color,
                            marginRight: 8,
                            borderRadius: 2
                          }}
                        />
                        <span>{cls.name}</span>
                        {selectedClass === index && (
                          <Tag color="green" size="small" style={{ marginLeft: 8 }}>
                            Active
                          </Tag>
                        )}
                      </div>
                    </List.Item>
                  )}
                />
              </Card>

              {/* Current Annotations */}
              <Card title="📝 Annotations" size="small">
                {annotations.length === 0 ? (
                  <div style={{ textAlign: 'center', color: '#999', padding: 20 }}>
                    <BulbOutlined style={{ fontSize: 24, marginBottom: 8 }} />
                    <div>No annotations yet</div>
                    <div style={{ fontSize: 12 }}>Use tools to start annotating</div>
                  </div>
                ) : (
                  <List
                    size="small"
                    dataSource={annotations}
                    renderItem={(annotation, index) => (
                      <List.Item>
                        <div style={{ width: '100%' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Space>
                              <div 
                                style={{
                                  width: 8,
                                  height: 8,
                                  backgroundColor: classes[annotation.classIndex]?.color || '#999',
                                  borderRadius: '50%'
                                }}
                              />
                              <span>{classes[annotation.classIndex]?.name || 'Unknown'}</span>
                              <Tag size="small">{annotation.type}</Tag>
                            </Space>
                            {annotation.confidence && (
                              <Tag color="blue" size="small">
                                {Math.round(annotation.confidence * 100)}%
                              </Tag>
                            )}
                          </div>
                          {annotation.source === 'ai' && (
                            <Tag color="purple" size="small" style={{ marginTop: 4 }}>
                              AI Generated
                            </Tag>
                          )}
                        </div>
                      </List.Item>
                    )}
                  />
                )}
              </Card>

              {/* Quick Tips */}
              <Card title="💡 Quick Tips" size="small">
                <div style={{ fontSize: 12, color: '#666' }}>
                  <div style={{ marginBottom: 8 }}>
                    <strong>Bounding Box:</strong> Click and drag to create
                  </div>
                  <div style={{ marginBottom: 8 }}>
                    <strong>Polygon:</strong> Click points, double-click to finish
                  </div>
                  <div style={{ marginBottom: 8 }}>
                    <strong>AI Segment:</strong> Click on object for instant segmentation
                  </div>
                  <div>
                    <strong>Pan:</strong> Hold and drag to move around
                  </div>
                </div>
              </Card>

            </Space>
          </div>
        </Col>
      </Row>

      {/* Hidden upload input */}
      <div style={{ display: 'none' }}>
        <Upload {...uploadProps}>
          <Button />
        </Upload>
      </div>

      {/* Add Class Modal */}
      <Modal
        title="Add New Class"
        open={showClassModal}
        onOk={handleAddClass}
        onCancel={() => {
          setShowClassModal(false);
          setNewClassName('');
        }}
        okText="Add Class"
      >
        <Input
          placeholder="Enter class name"
          value={newClassName}
          onChange={(e) => setNewClassName(e.target.value)}
          onPressEnter={handleAddClass}
        />
      </Modal>
    </div>
  );
};

export default Annotate;