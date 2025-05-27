"""
API routes for dataset management with full database integration
Handle image datasets, uploads, and auto-labeling
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from database.database import get_db
from database.operations import (
    DatasetOperations, ProjectOperations, ImageOperations, 
    AutoLabelJobOperations
)
from core.file_handler import file_handler
from core.auto_labeler import auto_labeler
from models.model_manager import model_manager

router = APIRouter()


class DatasetCreateRequest(BaseModel):
    """Request model for creating a new dataset"""
    name: str
    description: str = ""
    project_id: str
    auto_label_enabled: bool = True
    model_id: Optional[str] = None


class DatasetUpdateRequest(BaseModel):
    """Request model for updating a dataset"""
    name: Optional[str] = None
    description: Optional[str] = None
    auto_label_enabled: Optional[bool] = None
    model_id: Optional[str] = None


class AutoLabelRequest(BaseModel):
    """Request model for auto-labeling"""
    model_id: str
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    overwrite_existing: bool = False


@router.get("/", response_model=List[Dict[str, Any]])
async def get_datasets(
    project_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all datasets, optionally filtered by project"""
    try:
        if project_id:
            # Verify project exists
            project = ProjectOperations.get_project(db, project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            datasets = DatasetOperations.get_datasets_by_project(db, project_id)
        else:
            # Get all datasets
            datasets = DatasetOperations.get_all_datasets(db, skip=skip, limit=limit)
        
        dataset_responses = []
        for dataset in datasets:
            dataset_response = {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "project_id": dataset.project_id,
                "total_images": dataset.total_images,
                "labeled_images": dataset.labeled_images,
                "unlabeled_images": dataset.unlabeled_images,
                "auto_label_enabled": dataset.auto_label_enabled,
                "model_id": dataset.model_id,
                "created_at": dataset.created_at,
                "updated_at": dataset.updated_at
            }
            dataset_responses.append(dataset_response)
        
        return dataset_responses
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def create_dataset(
    request: DatasetCreateRequest,
    db: Session = Depends(get_db)
):
    """Create a new dataset"""
    try:
        # Verify project exists
        project = ProjectOperations.get_project(db, request.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Validate model_id if provided
        if request.model_id:
            model_info = model_manager.get_model_info(request.model_id)
            if not model_info:
                raise HTTPException(status_code=400, detail="Invalid model ID")
        
        # Create dataset
        dataset = DatasetOperations.create_dataset(
            db=db,
            name=request.name,
            project_id=request.project_id,
            description=request.description,
            auto_label_enabled=request.auto_label_enabled,
            model_id=request.model_id
        )
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "project_id": dataset.project_id,
            "total_images": dataset.total_images,
            "labeled_images": dataset.labeled_images,
            "unlabeled_images": dataset.unlabeled_images,
            "auto_label_enabled": dataset.auto_label_enabled,
            "model_id": dataset.model_id,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")


@router.post("/upload", response_model=Dict[str, Any])
async def upload_dataset(
    name: str = Form(...),
    description: str = Form(""),
    project_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    auto_label: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Create a new dataset and upload files to it"""
    try:
        # If no project_id provided, create a default project
        if not project_id:
            # Create a default project for this dataset
            default_project = ProjectOperations.create_project(
                db=db,
                name=f"Project for {name}",
                description=f"Auto-created project for dataset {name}"
            )
            project_id = default_project.id
        else:
            # Verify project exists
            project = ProjectOperations.get_project(db, project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        
        # Create dataset
        dataset = DatasetOperations.create_dataset(
            db=db,
            name=name,
            project_id=project_id,
            description=description,
            auto_label_enabled=auto_label,
            model_id=None  # Will be set during auto-labeling if enabled
        )
        
        # Upload files to the dataset
        upload_results = await file_handler.upload_images_to_dataset(
            files, dataset.id, auto_label=auto_label
        )
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "project_id": dataset.project_id,
            "total_images": len(files),
            "upload_results": upload_results,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")


@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Get a specific dataset with detailed information"""
    try:
        dataset = DatasetOperations.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get recent images
        recent_images = ImageOperations.get_images_by_dataset(
            db, dataset_id, skip=0, limit=10
        )
        
        image_list = []
        for image in recent_images:
            image_data = {
                "id": image.id,
                "filename": image.filename,
                "original_filename": image.original_filename,
                "width": image.width,
                "height": image.height,
                "file_size": image.file_size,
                "is_labeled": image.is_labeled,
                "is_auto_labeled": image.is_auto_labeled,
                "is_verified": image.is_verified,
                "created_at": image.created_at
            }
            image_list.append(image_data)
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "project_id": dataset.project_id,
            "total_images": dataset.total_images,
            "labeled_images": dataset.labeled_images,
            "unlabeled_images": dataset.unlabeled_images,
            "auto_label_enabled": dataset.auto_label_enabled,
            "model_id": dataset.model_id,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "recent_images": image_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {str(e)}")


@router.post("/{dataset_id}/upload")
async def upload_images(
    dataset_id: str,
    files: List[UploadFile] = File(...),
    auto_label: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Upload images to a dataset"""
    try:
        # Verify dataset exists
        dataset = DatasetOperations.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Upload images
        upload_results = await file_handler.upload_images_to_dataset(
            files, dataset_id, auto_label=auto_label
        )
        
        return upload_results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload images: {str(e)}")


@router.post("/{dataset_id}/auto-label")
async def start_auto_labeling(
    dataset_id: str,
    request: AutoLabelRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start auto-labeling job for a dataset"""
    try:
        # Verify dataset exists
        dataset = DatasetOperations.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify model exists
        model_info = model_manager.get_model_info(request.model_id)
        if not model_info:
            raise HTTPException(status_code=400, detail="Invalid model ID")
        
        # Create auto-label job
        job = AutoLabelJobOperations.create_auto_label_job(
            db=db,
            dataset_id=dataset_id,
            model_id=request.model_id,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            overwrite_existing=request.overwrite_existing
        )
        
        # Start auto-labeling in background
        background_tasks.add_task(
            auto_labeler.auto_label_dataset,
            dataset_id=dataset_id,
            model_id=request.model_id,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            overwrite_existing=request.overwrite_existing,
            job_id=job.id
        )
        
        return {
            "job_id": job.id,
            "message": "Auto-labeling job started",
            "dataset_id": dataset_id,
            "model_id": request.model_id,
            "status": "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start auto-labeling: {str(e)}")


@router.get("/{dataset_id}/images")
async def get_dataset_images(
    dataset_id: str,
    skip: int = 0,
    limit: int = 50,
    labeled_only: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get images in a dataset"""
    try:
        # Verify dataset exists
        dataset = DatasetOperations.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get images
        images = ImageOperations.get_images_by_dataset(
            db, dataset_id, skip=skip, limit=limit, labeled_only=labeled_only
        )
        
        image_list = []
        for image in images:
            image_data = {
                "id": image.id,
                "filename": image.filename,
                "original_filename": image.original_filename,
                "width": image.width,
                "height": image.height,
                "file_size": image.file_size,
                "is_labeled": image.is_labeled,
                "is_auto_labeled": image.is_auto_labeled,
                "is_verified": image.is_verified,
                "created_at": image.created_at,
                "url": file_handler.get_image_url(image.id)
            }
            image_list.append(image_data)
        
        return {
            "dataset_id": dataset_id,
            "images": image_list,
            "total_returned": len(image_list),
            "skip": skip,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset images: {str(e)}")


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset and all its images"""
    try:
        # Check if dataset exists
        dataset = DatasetOperations.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Clean up files
        file_handler.cleanup_dataset_files(dataset_id)
        
        # Delete dataset from database
        success = DatasetOperations.delete_dataset(db, dataset_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete dataset")
        
        return {"message": "Dataset deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.post("/upload")
async def upload_dataset(
    name: str = Form(...),
    description: str = Form(""),
    project_id: str = Form(...),
    auto_label_enabled: bool = Form(True),
    model_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Create a new dataset and upload images to it"""
    try:
        # Verify project exists
        project = ProjectOperations.get_project(db, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create dataset
        dataset_data = DatasetCreateRequest(
            name=name,
            description=description,
            project_id=project_id,
            auto_label_enabled=auto_label_enabled,
            model_id=model_id
        )
        
        dataset = DatasetOperations.create_dataset(db, dataset_data.dict())
        
        # Upload images to the new dataset
        upload_results = await file_handler.upload_images_to_dataset(
            dataset_id=dataset.id,
            files=files,
            auto_label=auto_label_enabled
        )
        
        return {
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "project_id": dataset.project_id,
                "auto_label_enabled": dataset.auto_label_enabled,
                "model_id": dataset.model_id,
                "created_at": dataset.created_at
            },
            "upload_results": upload_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")